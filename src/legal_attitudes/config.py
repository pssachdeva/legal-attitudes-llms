"""Pydantic config models for experiment runners."""
from pathlib import Path

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """A single prompt with its schema reference."""
    path: Path
    schema_name: str  # e.g., "OOLResponse"


class ModelConfig(BaseModel):
    """A single model to run."""
    provider: str  # openai, anthropic, google
    name: str      # model identifier


class BatchConfig(BaseModel):
    """Top-level batch experiment configuration."""
    experiment_name: str
    prompts: list[PromptConfig]
    models: list[ModelConfig]
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_completion_tokens: int = Field(default=500, ge=1)
    use_structured_output: bool = Field(default=True)
    seed: int | None = Field(default=None)
    repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=5, ge=1)

