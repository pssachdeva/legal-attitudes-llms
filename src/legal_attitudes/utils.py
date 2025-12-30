"""Utilities for project-wide paths, I/O, and logging."""
import json
import sys
from pathlib import Path

from loguru import logger


def find_root(markers=(".git", "pyproject.toml")) -> Path:
    """Return the project root by walking upward until a marker is found."""
    current = Path(__file__).resolve()
    for parent in (current,) + tuple(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError(f"Project root not found using markers: {markers}")


# Convenience paths anchored at the detected project root.
ROOT = find_root()
PROMPTS_DIR = ROOT / "prompts"
EXPERIMENTS_DIR = ROOT / "experiments"
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"


def save_result(experiment_name: str, scale_name: str, provider: str, model_name: str, data: dict) -> Path:
    """Save result data as JSON under RESULTS_DIR/{experiment}/{scale}_{provider}_{model}.json.

    Args:
        experiment_name: Subfolder inside results/ (from YAML config).
        scale_name: Name of the scale/prompt.
        provider: LLM provider name (e.g., "openai", "anthropic", "google").
        model_name: Model identifier; slashes and colons are sanitized.
        data: Dictionary to serialize as JSON.

    Returns:
        The full path where the file was written.
    """
    safe_model = model_name.replace("/", "_").replace(":", "_")
    filename = f"{scale_name}_{provider}_{safe_model}.json"
    full_path = RESULTS_DIR / experiment_name / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(data, indent=2))
    return full_path


def result_exists(experiment_name: str, scale_name: str, provider: str, model_name: str) -> bool:
    """Check if a result file already exists.

    Args:
        experiment_name: Subfolder inside results/ (from YAML config).
        scale_name: Name of the scale/prompt.
        provider: LLM provider name.
        model_name: Model identifier; slashes and colons are sanitized.

    Returns:
        True if the result file exists, False otherwise.
    """
    safe_model = model_name.replace("/", "_").replace(":", "_")
    filename = f"{scale_name}_{provider}_{safe_model}.json"
    full_path = RESULTS_DIR / experiment_name / filename
    return full_path.exists()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure loguru with a clean, readable format."""
    logger.remove()
    logger.add(
        sink=sys.stderr,
        format=(
            "<magenta><level>{level}</level></magenta>"
            " | "
            "<bold><cyan>{file}:{line}</cyan></bold>\n"
            "<bold>{message}</bold>\n<green><bold>"
            + "=" * 100
            + "</bold></green>"
        ),
    )

