"""Submit batch experiments to provider batch APIs for 50% discount.

Usage:
    python scripts/submit_batch.py experiments/exp_batch.yaml

Supports multiple models:
    - Each model in the config gets its own batch
    - All batch IDs and metadata saved to experiment folder
    - Process results with: python scripts/process_batch_results.py <config>
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from legal_attitudes.batch import (
    create_anthropic_batch,
    create_google_batch,
    create_openai_batch,
)
from legal_attitudes.config import BatchConfig
from legal_attitudes.utils import RESULTS_DIR, ROOT, setup_logging


def format_temperature_suffix(temperature: float) -> str:
    """Format temperature into a filename-safe suffix."""
    text = f"{temperature:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def model_temperature(model_cfg, default_temp: float) -> float:
    """Resolve per-model temperature with fallback to default."""
    return model_cfg.temperature if model_cfg.temperature is not None else default_temp


def model_id_for(model_cfg, default_temp: float) -> str:
    """Build a model identifier that encodes per-model temperature when set."""
    base = model_cfg.name
    if model_cfg.temperature is None:
        return base
    suffix = format_temperature_suffix(model_temperature(model_cfg, default_temp))
    return f"{base}__t{suffix}"


def save_batch_metadata(cfg: BatchConfig, batch_entries: list, config_path: Path):
    """Save batch metadata to experiment folder.

    Args:
        cfg: Experiment config
        batch_entries: List of batch info dicts, one per model
        config_path: Path to the config file
    """
    experiment_dir = RESULTS_DIR / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_name": cfg.experiment_name,
        "config_path": str(config_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "num_scales": len(cfg.prompts),
        "num_repeats": cfg.repeats,
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
    cfg = BatchConfig(**raw)

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Models: {len(cfg.models)}")
    logger.info(f"Scales: {len(cfg.prompts)}")
    logger.info(f"Repeats: {cfg.repeats}")
    logger.info(f"Requests per model: {len(cfg.prompts) * cfg.repeats}")
    logger.info(f"Total requests: {len(cfg.prompts) * cfg.repeats * len(cfg.models)}")

    # Load all prompts
    prompt_texts = {}
    for prompt_cfg in cfg.prompts:
        prompt_path = ROOT / prompt_cfg.path
        prompt_texts[prompt_path.stem] = prompt_path.read_text()

    # Setup experiment directory
    experiment_dir = RESULTS_DIR / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Submit a batch for each model
    batch_entries = []
    for model_cfg in cfg.models:
        provider = model_cfg.provider
        model_name = model_cfg.name
        model_temp = model_temperature(model_cfg, cfg.temperature)
        model_id = model_id_for(model_cfg, cfg.temperature)
        safe_model = model_id.replace("/", "_").replace(":", "_")

        logger.info("=" * 60)
        logger.info(f"Submitting batch for {provider}/{model_name} (temp={model_temp})")

        # Create a single-model config for this batch
        single_model_cfg = BatchConfig(
            experiment_name=cfg.experiment_name,
            prompts=cfg.prompts,
            models=[model_cfg],
            temperature=model_temp,
            max_completion_tokens=cfg.max_completion_tokens,
            seed=cfg.seed,
            use_structured_output=cfg.use_structured_output,
            repeats=cfg.repeats,
        )

        # Batch input path per model
        batch_input_path = experiment_dir / f"batch_input_{provider}_{safe_model}.jsonl"

        # Create and submit batch based on provider
        if provider == "openai":
            batch_info = create_openai_batch(single_model_cfg, prompt_texts, batch_input_path)
        elif provider == "anthropic":
            batch_info = create_anthropic_batch(single_model_cfg, prompt_texts, batch_input_path)
        elif provider == "google":
            batch_info = create_google_batch(single_model_cfg, prompt_texts, batch_input_path)
        else:
            raise ValueError(f"Unsupported provider for batch: {provider}")

        # Build batch entry
        batch_entry = {
            "batch_id": batch_info["batch_id"],
            "provider": provider,
            "model": model_name,
            "model_id": model_id,
            "temperature": model_temp,
            "status": batch_info["status"],
            "num_requests": batch_info["num_requests"],
        }

        # Add provider-specific fields
        if "input_file_id" in batch_info:
            batch_entry["input_file_id"] = batch_info["input_file_id"]
        if "file_name" in batch_info:
            batch_entry["file_name"] = batch_info["file_name"]

        batch_entries.append(batch_entry)

        logger.info(f"  Batch ID: {batch_info['batch_id']}")
        logger.info(f"  Status: {batch_info['status']}")

    # Save metadata with all batches
    metadata_path = save_batch_metadata(cfg, batch_entries, cfg_path)

    logger.info("=" * 60)
    logger.info(f"All {len(batch_entries)} batches submitted successfully!")
    logger.info(f"Metadata: {metadata_path}")
    logger.info("Next steps:")
    logger.info("1. Wait for batches to complete (check provider dashboard)")
    logger.info(f"2. Run: python scripts/process_batch_results.py {cfg_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Submit batch experiment to provider batch API"
    )
    parser.add_argument("config", help="Path to experiment YAML config")
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as exc:
        logger.error(f"Failed to submit batch: {exc}")
        raise
