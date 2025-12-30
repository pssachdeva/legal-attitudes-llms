"""API query functions for different LLM providers."""
import json

from anthropic import Anthropic
from google import genai
from openai import OpenAI


REFUSAL_CODE = 7

# Models that only support default temperature (1.0)
OPENAI_NO_TEMP_MODELS = {"gpt-5-2025-08-07"}


def query_openai(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output=True, seed=None):
    """Call OpenAI chat completions, optionally with JSON schema response format."""
    client = OpenAI()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_completion_tokens": max_tokens,
    }
    # Some models don't support custom temperature
    if model not in OPENAI_NO_TEMP_MODELS:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if use_structured_output:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema_cls.model_json_schema(),
                "strict": True,
            },
        }
    completion = client.chat.completions.create(**kwargs)
    raw_text = completion.choices[0].message.content
    extracted = extract_json(raw_text)
    if extracted is None:
        return {"raw": raw_text, "json": make_refusal_response(schema_cls)}
    return {"raw": raw_text, "json": extracted}


def extract_json(text: str) -> str | None:
    """Extract JSON object from text that may have markdown fences or trailing content.
    
    Returns None if no JSON object is found (likely a refusal).
    """
    # Strip markdown code fences if present.
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    # Find the JSON object boundaries.
    start = text.find("{")
    if start == -1:
        return None  # No JSON found - likely a refusal
    # Find matching closing brace.
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # Malformed JSON


def make_refusal_response(schema_cls) -> str:
    """Generate a JSON response with all values set to REFUSAL_CODE (9)."""
    # Get field aliases from the schema (the "1", "2", etc. keys)
    schema = schema_cls.model_json_schema()
    properties = schema.get("properties", {})
    refusal = {alias: REFUSAL_CODE for alias in properties.keys()}
    return json.dumps(refusal)


def query_anthropic(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output=True):
    """Call Anthropic messages API, instructing JSON output."""
    client = Anthropic()
    message = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_text = message.content[0].text
    extracted = extract_json(raw_text)
    if extracted is None:
        # Model refused to answer - return all 9s
        return {"raw": raw_text, "json": make_refusal_response(schema_cls)}
    return {"raw": raw_text, "json": extracted}


def query_google(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output=True):
    """Call Google Gemini with generation config."""
    client = genai.Client()
    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if use_structured_output:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema_cls
    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    raw_text = response.text
    # Gemini can return None if blocked/filtered - raise with reason
    if raw_text is None:
        reason = "unknown"
        if response.candidates:
            reason = str(response.candidates[0].finish_reason)
        elif response.prompt_feedback:
            reason = str(response.prompt_feedback.block_reason)
        raise RuntimeError(f"Gemini blocked response: {reason}")
    extracted = extract_json(raw_text)
    if extracted is None:
        raise RuntimeError(f"Gemini returned no JSON (possible refusal): {raw_text[:200]}")
    return {"raw": raw_text, "json": extracted}


def run_query(provider, model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output=True, seed=None):
    """Dispatch to the appropriate provider."""
    if provider == "openai":
        return query_openai(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output, seed)
    if provider == "anthropic":
        return query_anthropic(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output)
    if provider == "google":
        return query_google(model, prompt_text, temperature, max_tokens, schema_cls, use_structured_output)
    raise ValueError(f"Unsupported provider: {provider}")

