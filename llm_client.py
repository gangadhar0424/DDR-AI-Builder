"""
LLM Client Utility.

Provides a unified interface to call OpenAI, Anthropic Claude, or Google Gemini
APIs with retry logic, structured JSON extraction, and error handling.
"""

import json
import re

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

import config


# ── Non-retryable errors (fail fast) ──
_NON_RETRYABLE = []
try:
    from openai import AuthenticationError as OpenAIAuthError
    _NON_RETRYABLE.append(OpenAIAuthError)
except ImportError:
    pass
try:
    from anthropic import AuthenticationError as AnthropicAuthError
    _NON_RETRYABLE.append(AnthropicAuthError)
except ImportError:
    pass


def _should_retry(exc: BaseException) -> bool:
    """Return True only for transient errors. Auth errors fail immediately."""
    for non_retryable in _NON_RETRYABLE:
        if isinstance(exc, non_retryable):
            return False
    return True


def _call_openai(prompt: str, system_prompt: str, cfg: dict) -> str:
    """Call OpenAI chat completion API."""
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    return response.choices[0].message.content or ""


def _call_anthropic(prompt: str, system_prompt: str, cfg: dict) -> str:
    """Call Anthropic Claude messages API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=cfg["api_key"])
    kwargs = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    # Claude returns content as a list of blocks
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )


def _call_gemini(prompt: str, system_prompt: str, cfg: dict) -> str:
    """Call Google Gemini API via google-genai SDK."""
    from google import genai

    client = genai.Client(api_key=cfg["api_key"])

    # Build the full prompt with system instructions
    full_prompt = prompt
    generate_kwargs = {
        "model": cfg["model"],
        "contents": full_prompt,
    }

    # Add system instruction and generation config
    config_kwargs = {}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if cfg.get("temperature") is not None:
        config_kwargs["temperature"] = cfg["temperature"]
    if cfg.get("max_tokens"):
        config_kwargs["max_output_tokens"] = cfg["max_tokens"]

    if config_kwargs:
        generate_kwargs["config"] = genai.types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(**generate_kwargs)
    return response.text or ""


@retry(
    stop=stop_after_attempt(config.LLM_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=config.LLM_RETRY_DELAY, max=30),
    retry=retry_if_exception(_should_retry),
    before_sleep=lambda rs: logger.warning(
        f"LLM call failed (attempt {rs.attempt_number}), retrying..."
    ),
)
def call_llm(
    prompt: str,
    system_prompt: str = "",
    cfg: dict | None = None,
) -> str:
    """
    Call the configured LLM provider with retry logic.

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system/instruction prompt.
        cfg: Override LLM config. Defaults to global config.

    Returns:
        Raw text response from the LLM.
    """
    cfg = cfg or config.get_llm_config()
    provider = cfg.get("provider", "openai")

    logger.debug(f"Calling {provider} ({cfg['model']})")

    if provider == "anthropic":
        return _call_anthropic(prompt, system_prompt, cfg)
    elif provider == "gemini":
        return _call_gemini(prompt, system_prompt, cfg)
    else:
        return _call_openai(prompt, system_prompt, cfg)


def call_llm_json(
    prompt: str,
    system_prompt: str = "",
    cfg: dict | None = None,
) -> dict | list:
    """
    Call LLM and parse the response as JSON.

    Handles common issues like markdown code fences around JSON.

    Args:
        prompt: The user prompt (should request JSON output).
        system_prompt: Optional system prompt.
        cfg: Override LLM config.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ValueError: If response cannot be parsed as valid JSON.
    """
    raw = call_llm(prompt, system_prompt, cfg)
    return parse_json_response(raw)


def parse_json_response(raw: str) -> dict | list:
    """
    Parse LLM response text as JSON, handling common formatting.

    Strips markdown code fences, trailing commas, and other artifacts
    that LLMs often include in JSON responses.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the text cannot be parsed as JSON.
    """
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object in the text
    for pattern in [
        r"(\[[\s\S]*\])",  # JSON array
        r"(\{[\s\S]*\})",  # JSON object
    ]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Failed to parse JSON from LLM response:\n{raw[:500]}")
