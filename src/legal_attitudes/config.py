"""Pydantic config models for experiment runners."""
from pathlib import Path

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """A single prompt with its schema reference."""
    path: Path
    schema_name: str  # e.g., "OOLResponse"


class PromptMessagesConfig(BaseModel):
    """A prompt spec with optional system prompt and message file paths."""
    name: str
    schema_name: str  # e.g., "OOLResponse"
    system_prompt_path: Path | None = None
    persona_path: Path | None = None
    persona_name: str | None = None
    messages: list[Path]


class ModelConfig(BaseModel):
    """A single model to run."""
    provider: str  # openai, anthropic, google
    name: str      # model identifier
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None)
    use_structured_output: bool | None = Field(default=None)


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
    # Retry settings for async experiments
    max_retries: int = Field(default=10, ge=0)
    initial_backoff: float = Field(default=5.0, ge=0.1)


class MessageBatchConfig(BatchConfig):
    """Batch config that supports system prompt + YAML user messages."""
    system_prompt_path: Path | None = None


class BatchExperimentConfig(BaseModel):
    """Unified batch experiment configuration with system + message file paths."""
    experiment_name: str
    output_dir: Path | None = None
    persona_path: Path | None = None
    persona_name: str | None = None
    prompts: list[PromptMessagesConfig]
    models: list[ModelConfig]
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_completion_tokens: int = Field(default=500, ge=1)
    use_structured_output: bool = Field(default=True)
    seed: int | None = Field(default=None)
    repeats: int = Field(default=1, ge=1)
