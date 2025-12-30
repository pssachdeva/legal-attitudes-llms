"""Minimal prompt runner for local experiments.

Usage:
    python scripts/run_prompt.py [path/to/config.yaml]

The config YAML should at least provide a `prompt_path`. The script will read
the prompt file, call the model, and print the response. Requires
`OPENAI_API_KEY` in your environment.
"""
import argparse
from pathlib import Path

import yaml
from loguru import logger
from anthropic import Anthropic
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from legal_attitudes.utils import setup_logging

setup_logging()

class ExperimentConfig(BaseModel):
    """Typed settings for a single prompt run."""
    model_config = ConfigDict(populate_by_name=True)

    provider: str = Field(default="openai")
    model: str = Field(default="gpt-5.1")
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_completion_tokens: int = Field(default=400, ge=1, alias="max_tokens")
    prompt_path: Path
    seed: int | None = Field(default=None)


def load_config(path: str) -> ExperimentConfig:
    """Load and validate experiment settings from a YAML file."""
    cfg_path = Path(path).expanduser().resolve()
    # Read the YAML config and map it into the typed model.
    data = yaml.safe_load(cfg_path.read_text())
    return ExperimentConfig(**data)


def load_prompt(path: str) -> str:
    """Load prompt text from disk."""
    prompt_path = Path(path).expanduser()
    return prompt_path.read_text()


def run_query(cfg: ExperimentConfig) -> str:
    """Render the prompt, call the chat completion API, and return the reply text."""
    prompt_text = load_prompt(cfg.prompt_path)
    if cfg.provider == "openai":
        # Create a client using env-provided credentials (e.g., OPENAI_API_KEY).
        client = OpenAI()
        kwargs = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": cfg.temperature,
            "max_completion_tokens": cfg.max_completion_tokens,
        }
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    if cfg.provider == "anthropic":
        # Uses ANTHROPIC_API_KEY from the environment.
        client = Anthropic()
        message = client.messages.create(
            model=cfg.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_completion_tokens,
        )
        return message.content[0].text
    if cfg.provider == "google":
        # Uses GOOGLE_API_KEY from the environment.
        client = genai.Client()
        response = client.models.generate_content(
            model=cfg.model,
            contents=prompt_text,
            config={
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_completion_tokens,
            },
        )
        return response.text
    raise ValueError(f"Unsupported provider: {cfg.provider}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    logger.info(f"Using model={cfg.model}, prompt={cfg.prompt_path}")
    reply = run_query(cfg)
    logger.info(f"--- Response ---\n{reply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment config YAML")
    args = parser.parse_args()

    try:
        # Run the prompt with the supplied configuration.
        main(args.config)
    except ValidationError as exc:
        logger.error(f"Invalid config: {exc}")

