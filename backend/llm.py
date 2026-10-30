"""Thin wrapper around the optional AI model.

Every AI call in TruthTrace goes through `ask_json`. If no model is configured, or a
call fails, it returns None and the caller falls back to the deterministic heuristic
path. Callers must never treat model output as evidence on its own: anything the
model says about a source is re-checked against the retrieved text (see evidence.py).
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from . import config

log = logging.getLogger("truthtrace.llm")

_anthropic_client = None

SYSTEM_BASE = (
    "You are the analysis engine of TruthTrace, a news verification system. "
    "Rules: use ONLY the text provided to you; never invent sources, quotes, numbers, "
    "dates or URLs; when the provided text is not enough, say so with 'unknown' or "
    "'insufficient'. Keep facts and opinions separate. Influence and popularity are "
    "never evidence of truth. Reply with a single JSON object and nothing else."
)


def provider_info() -> dict:
    if config.LLM_PROVIDER == "anthropic":
        return {"provider": "anthropic", "model": config.ANTHROPIC_MODEL, "enabled": True}
    if config.LLM_PROVIDER == "openai_compat":
        return {"provider": "openai_compat", "model": config.LLM_MODEL, "enabled": True}
    return {"provider": "none", "model": None, "enabled": False}


def enabled() -> bool:
    return config.LLM_PROVIDER in ("anthropic", "openai_compat")


def _extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        if text[end - 1] in "}]":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return None


async def _ask_anthropic(system: str, prompt: str, max_tokens: int) -> str | None:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=120.0)
    try:
        response = await _anthropic_client.beta.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": config.ANTHROPIC_EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as exc:
        log.warning("Claude rate limited: %s", exc)
        return None
    except anthropic.APIStatusError as exc:
        log.warning("Claude API error %s: %s", exc.status_code, exc)
        return None
    except anthropic.APIConnectionError as exc:
        log.warning("Claude connection error: %s", exc)
        return None
    if response.stop_reason == "refusal":
        log.warning("Claude declined the request")
        return None
    return "".join(b.text for b in response.content if b.type == "text")


async def _ask_openai_compat(system: str, prompt: str, max_tokens: int) -> str | None:
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    body = {
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        log.warning("LLM call failed: %s", exc)
        return None


async def ask_json(task: str, prompt: str, max_tokens: int = 4000):
    """Ask the configured model for JSON. Returns parsed JSON or None."""
    if not enabled():
        return None
    system = SYSTEM_BASE + "\nTask: " + task
    if config.LLM_PROVIDER == "anthropic":
        text = await _ask_anthropic(system, prompt, max_tokens)
    else:
        text = await _ask_openai_compat(system, prompt, max_tokens)
    if not text:
        return None
    data = _extract_json(text)
    if data is None:
        log.warning("Model reply was not valid JSON for task %r", task[:60])
    return data
