"""Helpers for running repeated model queries via LiteLLM batch_completion and provider batch APIs."""
import json
import re
import time
from pathlib import Path

import yaml
from anthropic import Anthropic
from google import genai
from litellm import batch_completion
from loguru import logger
from openai import OpenAI

from legal_attitudes.api import OPENAI_NO_TEMP_MODELS, extract_json, make_refusal_response
from legal_attitudes.config import BatchConfig, BatchExperimentConfig, MessageBatchConfig
from legal_attitudes.schemas import get_schema


def run_repeats(
    prompt_text: str,
    model: str,
    schema_name: str,
    repeats: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 500,
    chunk_size: int = 5,
    max_retries: int = 10,
    initial_backoff: float = 5.0,
) -> list[dict]:
    """Run the same prompt N times using LiteLLM batch_completion with chunking and retries.

    Args:
        prompt_text: The prompt to send.
        model: LiteLLM model string (e.g., "openrouter/qwen/qwen3-max").
        schema_name: Name of the Pydantic schema for parsing/refusal.
        repeats: Number of times to run.
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        chunk_size: Number of requests to send concurrently per chunk.
        max_retries: Maximum retry attempts for failed requests.
        initial_backoff: Initial backoff time in seconds (doubles each retry).

    Returns:
        List of dicts with run_index, raw, json keys.
    """
    schema_cls = get_schema(schema_name)

    # Track results by their original index
    results: dict[int, dict] = {}
    pending_indices = list(range(repeats))

    retry_count = 0
    backoff = initial_backoff

    while pending_indices and retry_count <= max_retries:
        if retry_count > 0:
            logger.warning(
                f"Retry {retry_count}/{max_retries}: {len(pending_indices)} pending requests, "
                f"backing off {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)  # Cap at 2 minutes

        # Process in chunks
        still_pending = []
        for chunk_start in range(0, len(pending_indices), chunk_size):
            chunk_indices = pending_indices[chunk_start : chunk_start + chunk_size]

            # Build messages for this chunk
            messages = [[{"role": "user", "content": prompt_text}] for _ in chunk_indices]

            try:
                responses = batch_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                # Entire batch failed (e.g., connection error)
                logger.warning(f"Chunk failed with exception: {e}")
                still_pending.extend(chunk_indices)
                continue

            # Process each response in the chunk
            for idx, resp in zip(chunk_indices, responses):
                if isinstance(resp, Exception):
                    # Check if it's a rate limit error
                    err_str = str(resp).lower()
                    if "rate" in err_str or "429" in err_str or "limit" in err_str:
                        logger.debug(f"Rate limited on index {idx}")
                        still_pending.append(idx)
                    else:
                        # Non-rate-limit error - log and mark as refusal
                        logger.warning(f"Request {idx} failed: {resp}")
                        results[idx] = {
                            "run_index": idx,
                            "raw": f"ERROR: {resp}",
                            "json": make_refusal_response(schema_cls),
                        }
                else:
                    # Success - extract JSON
                    raw = resp.choices[0].message.content
                    extracted = extract_json(raw)
                    json_out = extracted if extracted else make_refusal_response(schema_cls)
                    results[idx] = {"run_index": idx, "raw": raw, "json": json_out}

            # Small delay between chunks to avoid hammering the API
            if chunk_start + chunk_size < len(pending_indices):
                time.sleep(0.5)

        pending_indices = still_pending
        if still_pending:
            retry_count += 1

    # Log if we gave up on some requests
    if pending_indices:
        logger.error(
            f"Gave up on {len(pending_indices)} requests after {max_retries} retries: {pending_indices}"
        )
        # Mark remaining as refusals
        for idx in pending_indices:
            results[idx] = {
                "run_index": idx,
                "raw": "ERROR: Max retries exceeded",
                "json": make_refusal_response(schema_cls),
            }

    # Return results sorted by original index
    return [results[i] for i in range(repeats)]


# ---------------------------------------------------------------------------
# Provider Batch API Submission Functions
# ---------------------------------------------------------------------------


def create_openai_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit OpenAI batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the JSONL batch input file

    Returns:
        dict with batch_id, input_file_id, status, and num_requests
    """
    client = OpenAI()
    model_cfg = cfg.models[0]

    # Build JSONL batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            # Build request body
            body = {
                "model": model_cfg.name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_completion_tokens": cfg.max_completion_tokens,
            }

            # Add temperature if model supports it
            from legal_attitudes.api import OPENAI_NO_TEMP_MODELS
            if model_cfg.name not in OPENAI_NO_TEMP_MODELS:
                body["temperature"] = cfg.temperature

            if cfg.seed is not None:
                body["seed"] = cfg.seed

            # Add structured output if requested
            if cfg.use_structured_output:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "schema": schema_cls.model_json_schema(),
                        "strict": True,
                    },
                }

            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

    # Write JSONL file
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    # Upload file to OpenAI
    with open(batch_input_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    logger.info(f"Uploaded batch file: {batch_input_file.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment_name": cfg.experiment_name,
            "num_scales": str(len(cfg.prompts)),
            "num_repeats": str(cfg.repeats),
        }
    )

    logger.info(f"Created OpenAI batch: {batch.id}")

    return {
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "num_requests": len(requests),
    }


def create_anthropic_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit Anthropic batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the JSONL batch input file (for reference)

    Returns:
        dict with batch_id, status, and num_requests
    """
    client = Anthropic()
    model_cfg = cfg.models[0]

    # Build batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model_cfg.name,
                    "max_tokens": cfg.max_completion_tokens,
                    "temperature": cfg.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt_text}]
                        }
                    ],
                }
            })

    logger.info(f"Submitting Anthropic batch with {len(requests)} requests")

    # Create batch (Anthropic accepts requests directly, no file upload needed)
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Created Anthropic batch: {batch.id}")

    # Save requests locally for reference
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    return {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "num_requests": len(requests),
    }


def create_google_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit Google (Gemini) batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the batch input file

    Returns:
        dict with batch_id, status, file_name, and num_requests
    """
    import os
    from google import genai
    from google.genai import types as gemini_types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable 'GEMINI_API_KEY' must be set for Gemini batching."
        )

    client = genai.Client(api_key=api_key)
    model_cfg = cfg.models[0]

    # Build JSONL batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            # Build Gemini batch request format
            request = {
                "key": custom_id,
                "request": {
                    "contents": [
                        {"parts": [{"text": prompt_text}], "role": "user"}
                    ],
                    "generation_config": {
                        "max_output_tokens": cfg.max_completion_tokens,
                        "temperature": cfg.temperature,
                    },
                },
            }

            # Add structured output if requested
            if cfg.use_structured_output:
                request["request"]["generation_config"]["response_mime_type"] = "application/json"
                request["request"]["generation_config"]["response_schema"] = schema_cls

            requests.append(request)

    # Write JSONL file
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    # Upload file to Gemini
    upload = client.files.upload(
        file=str(batch_input_path),
        config=gemini_types.UploadFileConfig(
            display_name=f"{cfg.experiment_name}_batch_input",
            mime_type="application/jsonl",
        ),
    )

    logger.info(f"Uploaded batch file: {upload.name}")

    # Create batch job
    batch_job = client.batches.create(
        model=model_cfg.name,
        src=upload.name,
    )

    logger.info(f"Created Gemini batch: {batch_job.name}")

    return {
        "batch_id": batch_job.name,
        "file_name": upload.name,
        "status": batch_job.state.name if hasattr(batch_job, 'state') else "SUBMITTED",
        "num_requests": len(requests),
    }


# ---------------------------------------------------------------------------
# System + multi-message user batch helpers
# ---------------------------------------------------------------------------


def load_system_prompt(path: Path | None) -> str | None:
    """Load system prompt text from disk, if provided."""
    if path is None:
        return None
    return path.expanduser().read_text()


def load_message_files(paths: list[Path]) -> list[str]:
    """Load a list of message files in order."""
    return [path.expanduser().read_text() for path in paths]


def _ordered_message_keys(data: dict) -> list[str]:
    """Return message keys ordered by numeric suffix when present."""
    message_keys = [k for k in data.keys() if k.startswith("message")]
    if not message_keys:
        return []
    numeric = []
    non_numeric = []
    for key in message_keys:
        match = re.fullmatch(r"message(\d+)", key)
        if match:
            numeric.append((int(match.group(1)), key))
        else:
            non_numeric.append(key)
    if numeric:
        numeric.sort(key=lambda item: item[0])
        return [key for _, key in numeric] + non_numeric
    return message_keys


def load_user_messages(path: Path) -> list[str]:
    """Load user messages from a YAML file.

    Supported formats:
      - List of strings:
          - "message one"
          - "message two"
      - Dict with message1/message2 keys:
          message1: "message one"
          message2: "message two"
      - Dict with `messages` list containing strings or {content, role} objects
    """
    data = yaml.safe_load(path.read_text())
    if data is None:
        return []

    if isinstance(data, list):
        raw_messages = data
    elif isinstance(data, dict):
        if "messages" in data:
            raw_messages = data["messages"]
        else:
            keys = _ordered_message_keys(data)
            if not keys:
                raise ValueError(
                    f"{path} must include a 'messages' list or message1/message2 keys."
                )
            raw_messages = [data[k] for k in keys]
    else:
        raise ValueError(f"{path} must be a list or mapping.")

    messages: list[str] = []
    for idx, item in enumerate(raw_messages, start=1):
        if isinstance(item, str):
            messages.append(item)
            continue
        if isinstance(item, dict):
            role = item.get("role", "user")
            if role != "user":
                raise ValueError(
                    f"{path} message {idx} has role '{role}'. Only 'user' is supported."
                )
            if "content" in item:
                messages.append(item["content"])
                continue
            if "text" in item:
                messages.append(item["text"])
                continue
            if "message" in item:
                messages.append(item["message"])
                continue
            raise ValueError(
                f"{path} message {idx} must include 'content', 'text', or 'message'."
            )
        raise ValueError(f"{path} message {idx} must be a string or mapping.")

    return messages


def build_openai_messages(system_prompt: str | None, user_messages: list[str]) -> list[dict]:
    """Build OpenAI-style message list with optional system prompt."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in user_messages:
        messages.append({"role": "user", "content": msg})
    return messages


def build_anthropic_messages(user_messages: list[str]) -> list[dict]:
    """Build Anthropic-style message list (system handled separately)."""
    return [
        {"role": "user", "content": [{"type": "text", "text": msg}]}
        for msg in user_messages
    ]


def build_google_contents(user_messages: list[str]) -> list[dict]:
    """Build Gemini-style contents list (system handled separately)."""
    return [
        {"parts": [{"text": msg}], "role": "user"}
        for msg in user_messages
    ]


def build_openai_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
) -> list[dict]:
    """Build OpenAI batch requests for system + user messages."""
    model_cfg = cfg.models[0]
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = _prompt_name(prompt_cfg)
            user_messages = prompt_messages[prompt_name]
            system_prompt = system_prompts.get(prompt_name)
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"
            body = {
                "model": model_cfg.name,
                "messages": build_openai_messages(system_prompt, user_messages),
                "max_completion_tokens": cfg.max_completion_tokens,
            }
            if model_cfg.name not in OPENAI_NO_TEMP_MODELS:
                body["temperature"] = cfg.temperature
            if cfg.seed is not None:
                body["seed"] = cfg.seed
            if cfg.use_structured_output:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "schema": schema_cls.model_json_schema(),
                        "strict": True,
                    },
                }
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })
    return requests


def submit_openai_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    requests: list[dict],
    batch_input_path: Path,
) -> dict:
    """Submit OpenAI batch requests and return metadata."""
    client = OpenAI()

    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    with open(batch_input_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    logger.info(f"Uploaded batch file: {batch_input_file.id}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment_name": cfg.experiment_name,
            "num_scales": str(len(cfg.prompts)),
            "num_repeats": str(cfg.repeats),
        },
    )

    logger.info(f"Created OpenAI batch: {batch.id}")
    return {
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "num_requests": len(requests),
    }


def _prompt_name(prompt_cfg) -> str:
    """Resolve prompt name from config (supports multiple config shapes)."""
    if hasattr(prompt_cfg, "name"):
        return prompt_cfg.name
    return prompt_cfg.path.stem


def create_openai_batch_messages(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
    batch_input_path: Path,
) -> dict:
    """Create and submit OpenAI batch request with system + user messages."""
    requests = build_openai_batch_requests(cfg, prompt_messages, system_prompts)
    return submit_openai_batch_requests(cfg, requests, batch_input_path)


def build_anthropic_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
) -> list[dict]:
    """Build Anthropic batch requests for system + user messages."""
    model_cfg = cfg.models[0]
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = _prompt_name(prompt_cfg)
            user_messages = prompt_messages[prompt_name]
            system_prompt = system_prompts.get(prompt_name)
            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            params = {
                "model": model_cfg.name,
                "max_tokens": cfg.max_completion_tokens,
                "temperature": cfg.temperature,
                "messages": build_anthropic_messages(user_messages),
            }
            if system_prompt:
                params["system"] = system_prompt

            requests.append({
                "custom_id": custom_id,
                "params": params,
            })
    return requests


def submit_anthropic_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    requests: list[dict],
    batch_input_path: Path,
) -> dict:
    """Submit Anthropic batch requests and return metadata."""
    client = Anthropic()

    logger.info(f"Submitting Anthropic batch with {len(requests)} requests")
    batch = client.messages.batches.create(requests=requests)
    logger.info(f"Created Anthropic batch: {batch.id}")

    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    return {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "num_requests": len(requests),
    }


def create_anthropic_batch_messages(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
    batch_input_path: Path,
) -> dict:
    """Create and submit Anthropic batch request with system + user messages."""
    requests = build_anthropic_batch_requests(cfg, prompt_messages, system_prompts)
    return submit_anthropic_batch_requests(cfg, requests, batch_input_path)


def build_google_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
) -> list[dict]:
    """Build Google batch requests for system + user messages."""
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = _prompt_name(prompt_cfg)
            user_messages = prompt_messages[prompt_name]
            system_prompt = system_prompts.get(prompt_name)
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"
            request = {
                "key": custom_id,
                "request": {
                    "contents": build_google_contents(user_messages),
                    "generation_config": {
                        "max_output_tokens": cfg.max_completion_tokens,
                        "temperature": cfg.temperature,
                    },
                },
            }
            if system_prompt:
                request["request"]["system_instruction"] = {
                    "parts": [{"text": system_prompt}]
                }
            if cfg.use_structured_output:
                request["request"]["generation_config"]["response_mime_type"] = "application/json"
                request["request"]["generation_config"]["response_schema"] = schema_cls

            requests.append(request)
    return requests


def submit_google_batch_requests(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    requests: list[dict],
    batch_input_path: Path,
) -> dict:
    """Submit Google (Gemini) batch requests and return metadata."""
    import os
    from google.genai import types as gemini_types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable 'GEMINI_API_KEY' must be set for Gemini batching."
        )

    client = genai.Client(api_key=api_key)
    model_cfg = cfg.models[0]

    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    upload = client.files.upload(
        file=str(batch_input_path),
        config=gemini_types.UploadFileConfig(
            display_name=f"{cfg.experiment_name}_batch_input",
            mime_type="application/jsonl",
        ),
    )

    logger.info(f"Uploaded batch file: {upload.name}")

    batch_job = client.batches.create(
        model=model_cfg.name,
        src=upload.name,
    )

    logger.info(f"Created Gemini batch: {batch_job.name}")

    return {
        "batch_id": batch_job.name,
        "file_name": upload.name,
        "status": batch_job.state.name if hasattr(batch_job, "state") else "SUBMITTED",
        "num_requests": len(requests),
    }


def create_google_batch_messages(
    cfg: BatchExperimentConfig | MessageBatchConfig,
    prompt_messages: dict[str, list[str]],
    system_prompts: dict[str, str | None],
    batch_input_path: Path,
) -> dict:
    """Create and submit Google (Gemini) batch request with system + user messages."""
    requests = build_google_batch_requests(cfg, prompt_messages, system_prompts)
    return submit_google_batch_requests(cfg, requests, batch_input_path)
