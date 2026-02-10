"""Utilities for parsing batch outputs into result files and dataframes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from legal_attitudes.api import extract_json, make_refusal_response
from legal_attitudes.config import BatchExperimentConfig
from legal_attitudes.schemas import get_schema


def build_scale_to_persona(cfg: BatchExperimentConfig) -> dict[str, str]:
    """Build a map of prompt/scale name to effective persona name.

    Precedence:
      1) prompt-level persona_name
      2) experiment-level persona_name
      3) omitted
    """
    scale_to_persona: dict[str, str] = {}
    for prompt_cfg in cfg.prompts:
        effective_name = prompt_cfg.persona_name or cfg.persona_name
        if effective_name:
            scale_to_persona[prompt_cfg.name] = effective_name
    return scale_to_persona


def parse_and_save_results(
    output_path: Path,
    experiment_dir: Path,
    provider: str,
    model: str,
    cfg: BatchExperimentConfig,
    force: bool = False,
    model_id: str | None = None,
    temperature: float | None = None,
) -> None:
    """Parse batch output JSONL and save individual result files."""
    schema_map = {p.name: p.schema_name for p in cfg.prompts}

    model_key = model_id or model
    safe_model = model_key.replace("/", "_").replace(":", "_")
    model_dir = experiment_dir / f"{provider}_{safe_model}"
    model_dir.mkdir(parents=True, exist_ok=True)

    results_parsed = 0
    results_skipped = 0
    results_failed = 0

    with open(output_path) as f:
        for line in f:
            result = json.loads(line)

            if provider == "openai":
                custom_id = result.get("custom_id")
                response = result.get("response", {})
                if response.get("status_code") != 200:
                    logger.error(f"Failed request: {custom_id} - {response.get('body', {}).get('error')}")
                    results_failed += 1
                    continue
                raw_text = response["body"]["choices"][0]["message"]["content"]

            elif provider == "anthropic":
                custom_id = result.get("custom_id")
                if result.get("result", {}).get("type") == "error":
                    logger.error(f"Failed request: {custom_id} - {result['result']['error']}")
                    results_failed += 1
                    continue
                raw_text = result["result"]["message"]["content"][0]["text"]

            elif provider == "google":
                custom_id = result.get("key")
                response = result.get("response")
                if not response or "error" in response:
                    logger.error(f"Failed request: {custom_id} - {response.get('error') if response else 'no response'}")
                    results_failed += 1
                    continue

                try:
                    candidates = response.get("candidates", [])
                    if not candidates:
                        logger.error(f"No candidates in response for {custom_id}: {response}")
                        results_failed += 1
                        continue

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])

                    if parts:
                        raw_text = parts[0].get("text", "")
                    elif "text" in content:
                        raw_text = content["text"]
                    else:
                        logger.error(f"Could not extract text from response for {custom_id}: {response}")
                        results_failed += 1
                        continue

                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error parsing Google response for {custom_id}: {e}")
                    logger.debug(f"Response structure: {response}")
                    results_failed += 1
                    continue

            else:
                raise ValueError(f"Unknown provider: {provider}")

            parts = custom_id.rsplit("_repeat_", 1)
            if len(parts) != 2:
                logger.error(f"Invalid custom_id format: {custom_id}")
                results_failed += 1
                continue

            scale_name = parts[0]
            repeat_num = int(parts[1])

            output_file = model_dir / f"{scale_name}_repeat_{repeat_num:03d}.json"
            if output_file.exists() and not force:
                results_skipped += 1
                continue

            schema_name = schema_map.get(scale_name)
            if not schema_name:
                logger.error(f"Unknown scale: {scale_name}")
                results_failed += 1
                continue

            schema_cls = get_schema(schema_name)

            extracted = extract_json(raw_text)
            if extracted is None:
                logger.warning(f"Failed to extract JSON from {custom_id}, marking as refusal")
                json_out = make_refusal_response(schema_cls)
            else:
                json_out = extracted

            result_obj = {
                "run_index": repeat_num - 1,
                "prompt": scale_name,
                "provider": provider,
                "model": model,
                "temperature": temperature if temperature is not None else cfg.temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": raw_text,
                "json": json_out,
            }

            output_file.write_text(json.dumps(result_obj, indent=2))
            results_parsed += 1

    logger.info(f"Results: {results_parsed} saved, {results_skipped} skipped, {results_failed} failed")


def create_results_dataframe(
    experiment_dir: Path,
    provider: str,
    model: str,
    temperature: float,
    model_id: str | None = None,
    scale_to_persona: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a tidy dataframe from all result files."""
    model_key = model_id or model
    safe_model = model_key.replace("/", "_").replace(":", "_")
    model_dir = experiment_dir / f"{provider}_{safe_model}"

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return pd.DataFrame()

    rows = []
    include_persona = bool(scale_to_persona)

    for result_file in sorted(model_dir.glob("*_repeat_*.json")):
        filename = result_file.stem
        parts = filename.rsplit("_repeat_", 1)
        if len(parts) != 2:
            logger.warning(f"Skipping file with unexpected name: {result_file}")
            continue

        scale_name = parts[0]
        repeat_num = int(parts[1])

        try:
            result = json.loads(result_file.read_text())
            json_field = result["json"]
            raw_output = result.get("raw", "")

            if isinstance(json_field, dict):
                parsed_json = json_field
            elif isinstance(json_field, str):
                parsed_json = json.loads(json_field)
            else:
                logger.warning(f"Unexpected json type in {result_file}: {type(json_field)}")
                continue

            for item_num, response_value in parsed_json.items():
                row = {
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                    "scale": scale_name,
                    "repeat": repeat_num,
                    "item": item_num,
                    "response": response_value,
                    "raw_output": raw_output,
                }
                if include_persona:
                    row["persona"] = scale_to_persona.get(scale_name)
                rows.append(row)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {result_file.name} (likely placeholder/refusal) - skipping")
            continue
        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
            continue

    df = pd.DataFrame(rows)
    logger.info(f"Created dataframe with {len(df)} rows ({len(rows)} responses)")
    return df
