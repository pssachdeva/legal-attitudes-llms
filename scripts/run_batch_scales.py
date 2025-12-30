"""Batch runner for legal attitude scale experiments.

Usage:
    python scripts/run_batch_scales.py experiments/batch_config.yaml

Runs a matrix of prompts × models, saves structured JSON results, and skips
already-completed runs for resumability.
"""
import argparse
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from legal_attitudes.api import run_query
from legal_attitudes.config import BatchConfig
from legal_attitudes.schemas import get_schema
from legal_attitudes.utils import result_exists, save_result, setup_logging

setup_logging()

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def load_prompt(path: Path) -> str:
    """Load prompt text from disk."""
    return path.expanduser().read_text()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_batch(config_path: str):
    """Load batch config and run all prompt × model combinations."""
    cfg_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = BatchConfig(**raw)

    total = len(cfg.prompts) * len(cfg.models)
    completed = 0
    skipped = 0

    for prompt_cfg in cfg.prompts:
        prompt_name = prompt_cfg.path.stem
        prompt_text = load_prompt(prompt_cfg.path)
        schema_cls = get_schema(prompt_cfg.schema_name)

        for model_cfg in cfg.models:
            # Resumability: skip if result exists.
            if result_exists(cfg.experiment_name, prompt_name, model_cfg.provider, model_cfg.name):
                logger.info(f"[skip] {prompt_name} / {model_cfg.provider}:{model_cfg.name} (already done)")
                skipped += 1
                continue

            logger.info(f"[run] {prompt_name} / {model_cfg.provider}:{model_cfg.name}")
            try:
                response = run_query(
                    provider=model_cfg.provider,
                    model=model_cfg.name,
                    prompt_text=prompt_text,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_completion_tokens,
                    schema_cls=schema_cls,
                    use_structured_output=cfg.use_structured_output,
                    seed=cfg.seed,
                )
                # Validate and parse the JSON response.
                parsed = schema_cls.model_validate_json(response["json"])
                result = {
                    "prompt": str(prompt_cfg.path),
                    "provider": model_cfg.provider,
                    "model": model_cfg.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "raw_response": response["raw"],
                    "parsed": parsed.model_dump(by_alias=True),
                }
                out_path = save_result(cfg.experiment_name, prompt_name, model_cfg.provider, model_cfg.name, result)
                logger.info(f"[done] saved to {out_path}")
                completed += 1
            except Exception as exc:
                logger.error(f"[fail] {prompt_name} / {model_cfg.provider}:{model_cfg.name}: {exc}")

    logger.info(f"Batch complete: {completed} completed, {skipped} skipped, {total - completed - skipped} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch legal scale experiments")
    parser.add_argument("config", help="Path to batch config YAML")
    args = parser.parse_args()

    try:
        run_batch(args.config)
    except ValidationError as exc:
        logger.error(f"Invalid config: {exc}")

