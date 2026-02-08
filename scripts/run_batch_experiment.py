"""Submit batch experiments with optional system prompts and message file paths.

Usage:
    python scripts/run_batch_experiment.py experiments/exp_batch.yaml

Config example:
  experiment_name: my_exp
  output_dir: results/my_exp
  persona_path: prompts/personas/default.txt
  temperature: 0.2
  max_completion_tokens: 400
  repeats: 5
  prompts:
    - name: compliance
      schema_name: OOLResponse
      system_prompt_path: prompts/system/persona.txt
      persona_path: prompts/personas/compliance.txt
      messages:
        - prompts/messages/compliance_01.txt
        - prompts/messages/compliance_02.txt
  models:
    - provider: openai
      name: gpt-5.1
      temperature: 0.0
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from legal_attitudes.batch import (
    create_anthropic_batch_messages,
    create_google_batch_messages,
    create_openai_batch_messages,
    load_message_files,
    load_system_prompt,
)
from legal_attitudes.config import BatchExperimentConfig
from legal_attitudes.utils import RESULTS_DIR, ROOT, setup_logging


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against project root."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def compose_system_prompt(system_text: str | None, persona_text: str | None) -> str | None:
    """Combine system text and persona text with a blank line separator."""
    if system_text and persona_text:
        return f"{system_text}\n\n{persona_text}"
    if system_text:
        return system_text
    if persona_text:
        return persona_text
    return None


def resolve_prompt_persona_path(cfg: BatchExperimentConfig, prompt_cfg) -> Path | None:
    """Resolve effective persona path with prompt-level override precedence."""
    if prompt_cfg.persona_path:
        return prompt_cfg.persona_path
    return cfg.persona_path


def format_temperature_suffix(temperature: float) -> str:
    """Format temperature into a filename-safe suffix."""
    text = f"{temperature:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def model_temperature(model_cfg, default_temp: float) -> float:
    """Resolve per-model temperature with fallback to default."""
    return model_cfg.temperature if model_cfg.temperature is not None else default_temp


def model_max_tokens(model_cfg, default_tokens: int) -> int:
    """Resolve per-model max tokens with fallback to default."""
    return model_cfg.max_completion_tokens if model_cfg.max_completion_tokens is not None else default_tokens


def model_seed(model_cfg, default_seed: int | None) -> int | None:
    """Resolve per-model seed with fallback to default."""
    return model_cfg.seed if model_cfg.seed is not None else default_seed


def model_use_structured_output(model_cfg, default_value: bool) -> bool:
    """Resolve per-model structured output flag with fallback to default."""
    return model_cfg.use_structured_output if model_cfg.use_structured_output is not None else default_value


def model_id_for(model_cfg, default_temp: float) -> str:
    """Build a model identifier that encodes per-model temperature when set."""
    base = model_cfg.name
    if model_cfg.temperature is None:
        return base
    suffix = format_temperature_suffix(model_temperature(model_cfg, default_temp))
    return f"{base}__t{suffix}"


def resolve_experiment_dir(cfg: BatchExperimentConfig) -> Path:
    if cfg.output_dir:
        out_dir = Path(cfg.output_dir).expanduser()
        if not out_dir.is_absolute():
            return ROOT / out_dir
        return out_dir
    return RESULTS_DIR / cfg.experiment_name


def save_batch_metadata(cfg: BatchExperimentConfig, batch_entries: list, config_path: Path, experiment_dir: Path):
    """Save batch metadata to experiment folder."""
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_name": cfg.experiment_name,
        "config_path": str(config_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "num_prompts": len(cfg.prompts),
        "num_repeats": cfg.repeats,
        "output_dir": str(cfg.output_dir) if cfg.output_dir else None,
        "persona_path": str(cfg.persona_path) if cfg.persona_path else None,
        "prompts": [
            {
                "name": p.name,
                "schema_name": p.schema_name,
                "system_prompt_path": str(p.system_prompt_path) if p.system_prompt_path else None,
                "persona_path": str(p.persona_path) if p.persona_path else None,
                "messages": [str(m) for m in p.messages],
            }
            for p in cfg.prompts
        ],
        "batches": batch_entries,
    }

    metadata_path = experiment_dir / "batch_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(f"Saved batch metadata to {metadata_path}")
    return metadata_path


def main(config_path: str):
    """Load config, create batches for each model, submit, and save metadata."""
    cfg_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = BatchExperimentConfig(**raw)

    experiment_dir = resolve_experiment_dir(cfg)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Models: {len(cfg.models)}")
    logger.info(f"Prompts: {len(cfg.prompts)}")
    logger.info(f"Repeats: {cfg.repeats}")
    logger.info(f"Requests per model: {len(cfg.prompts) * cfg.repeats}")
    logger.info(f"Total requests: {len(cfg.prompts) * cfg.repeats * len(cfg.models)}")
    logger.info(f"Output dir: {experiment_dir}")

    # Load all prompt messages and effective system prompts.
    # Effective system prompt = (system prompt text) + optional persona text.
    prompt_messages: dict[str, list[str]] = {}
    system_prompts: dict[str, str | None] = {}
    for prompt_cfg in cfg.prompts:
        prompt_name = prompt_cfg.name
        system_text = (
            load_system_prompt(resolve_path(prompt_cfg.system_prompt_path))
            if prompt_cfg.system_prompt_path
            else None
        )
        persona_path = resolve_prompt_persona_path(cfg, prompt_cfg)
        persona_text = (
            load_system_prompt(resolve_path(persona_path))
            if persona_path
            else None
        )
        system_prompts[prompt_name] = compose_system_prompt(system_text, persona_text)
        message_paths = [resolve_path(p) for p in prompt_cfg.messages]
        prompt_messages[prompt_name] = load_message_files(message_paths)

    # Submit a batch for each model
    batch_entries = []
    for model_cfg in cfg.models:
        provider = model_cfg.provider
        model_name = model_cfg.name
        model_temp = model_temperature(model_cfg, cfg.temperature)
        model_tokens = model_max_tokens(model_cfg, cfg.max_completion_tokens)
        model_seed_value = model_seed(model_cfg, cfg.seed)
        model_structured = model_use_structured_output(model_cfg, cfg.use_structured_output)
        model_id = model_id_for(model_cfg, cfg.temperature)
        safe_model = model_id.replace("/", "_").replace(":", "_")

        logger.info("=" * 60)
        logger.info(f"Submitting batch for {provider}/{model_name} (temp={model_temp})")

        # Create a single-model config for this batch
        single_model_cfg = BatchExperimentConfig(
            experiment_name=cfg.experiment_name,
            output_dir=cfg.output_dir,
            prompts=cfg.prompts,
            models=[model_cfg],
            temperature=model_temp,
            max_completion_tokens=model_tokens,
            seed=model_seed_value,
            use_structured_output=model_structured,
            repeats=cfg.repeats,
        )

        batch_input_path = experiment_dir / f"batch_input_{provider}_{safe_model}.jsonl"

        if provider == "openai":
            batch_info = create_openai_batch_messages(
                single_model_cfg,
                prompt_messages,
                system_prompts,
                batch_input_path,
            )
        elif provider == "anthropic":
            batch_info = create_anthropic_batch_messages(
                single_model_cfg,
                prompt_messages,
                system_prompts,
                batch_input_path,
            )
        elif provider == "google":
            batch_info = create_google_batch_messages(
                single_model_cfg,
                prompt_messages,
                system_prompts,
                batch_input_path,
            )
        else:
            raise ValueError(f"Unsupported provider for batch: {provider}")

        batch_entry = {
            "batch_id": batch_info["batch_id"],
            "provider": provider,
            "model": model_name,
            "model_id": model_id,
            "temperature": model_temp,
            "max_completion_tokens": model_tokens,
            "status": batch_info["status"],
            "num_requests": batch_info["num_requests"],
        }
        if "input_file_id" in batch_info:
            batch_entry["input_file_id"] = batch_info["input_file_id"]
        if "file_name" in batch_info:
            batch_entry["file_name"] = batch_info["file_name"]

        batch_entries.append(batch_entry)

        logger.info(f"  Batch ID: {batch_info['batch_id']}")
        logger.info(f"  Status: {batch_info['status']}")

    metadata_path = save_batch_metadata(cfg, batch_entries, cfg_path, experiment_dir)

    logger.info("=" * 60)
    logger.info(f"All {len(batch_entries)} batches submitted successfully!")
    logger.info(f"Metadata: {metadata_path}")
    logger.info("Next steps:")
    logger.info("1. Wait for batches to complete (check provider dashboard)")
    logger.info(f"2. Run: python scripts/process_batch_results.py {cfg_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Submit batch experiment to provider batch API with system+message files"
    )
    parser.add_argument("config", help="Path to experiment YAML config")
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as exc:
        logger.error(f"Failed to submit batch: {exc}")
        raise
