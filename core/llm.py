"""Shared Google Gemini client. The key is read from env or Streamlit secrets, never hardcoded."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gemini-4-hydrogen"

FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
)

# Ensure thinking-enabled Gemini models retain output budget after reasoning tokens.
_MIN_OUTPUT_TOKENS = 2048

_clients: dict[str, genai.Client] = {}


def _read_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        secrets = st.secrets
        if name in secrets and secrets[name]:
            return str(secrets[name]).strip()

        lowered = name.lower()
        for key in secrets:
            val = secrets[key]
            if str(key).lower() == lowered and isinstance(val, (str, int, float)):
                return str(val).strip()
            if hasattr(val, "items"):
                for sub_key, sub_val in val.items():
                    if str(sub_key).lower() == lowered and sub_val:
                        return str(sub_val).strip()
        return ""
    except Exception:
        return ""


def _get_setting(name: str) -> str:
    env_val = (os.getenv(name) or "").strip()
    if env_val:
        return env_val
    return _read_streamlit_secret(name)


def require_api_key() -> str:
    key = _get_setting("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file in the project folder "
            "or set GEMINI_API_KEY in Streamlit secrets."
        )
    return key


def _model_order() -> list[str]:
    preferred = _get_setting("GEMINI_MODEL") or DEFAULT_MODEL
    ordered = [preferred, *FALLBACK_MODELS]
    seen = set()
    unique = []

    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)

    return unique


def _client(api_key: str) -> genai.Client:
    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def _content_to_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()

    return str(content or "").strip()


def _messages_to_gemini(value) -> tuple[str | None, list[types.Content] | str]:
    messages = value.to_messages() if hasattr(value, "to_messages") else value
    if isinstance(messages, str):
        return None, messages

    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "human")
        raw_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        text = _content_to_text(raw_content)
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
        elif role in {"ai", "assistant", "model"}:
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

    return system_instruction, contents


def _response_text(response) -> str:
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None

    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))

    return "".join(parts).strip()


def _classify_api_error(exc: Exception) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    raw_msg = str(getattr(exc, "message", "") or exc).strip()
    lowered = raw_msg.lower()

    if (
        code in (401, 403)
        or status in {"UNAUTHENTICATED", "PERMISSION_DENIED"}
        or any(tok in lowered for tok in ("api key not valid", "api_key_invalid", "invalid api key", "permission denied", "unauthenticated"))
    ):
        return (
            "auth",
            "Your Gemini API key is invalid or unauthorized. Verify GEMINI_API_KEY in .env or Streamlit secrets.",
        )

    if (
        code == 429
        or status == "RESOURCE_EXHAUSTED"
        or any(tok in lowered for tok in ("429", "quota", "rate limit", "resource_exhausted", "too many requests"))
    ):
        return (
            "quota",
            "Gemini API quota or rate limit exceeded. Wait a moment and retry, or check your Google AI Studio usage limits.",
        )

    if (
        code == 404
        or status == "NOT_FOUND"
        or any(tok in lowered for tok in ("not found", "is not supported for generatecontent", "unknown model"))
    ):
        return (
            "model",
            f"The Gemini model is unavailable or unsupported ({raw_msg}).",
        )

    return ("api", f"Gemini API error: {raw_msg}")


def get_llm(max_tokens: int = 1200):
    """Runnable that tries the preferred Gemini model, then supported fallbacks."""

    def _call(value):
        client = _client(require_api_key())
        system_instruction, contents = _messages_to_gemini(value)
        effective_max_tokens = max(max_tokens, _MIN_OUTPUT_TOKENS)
        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=effective_max_tokens,
            system_instruction=system_instruction,
        )

        categories: list[str] = []
        errors: list[str] = []

        for model in _model_order():
            print(f"[LLM] Invoking Gemini model={model} max_output_tokens={effective_max_tokens}")
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except APIError as exc:
                category, clean_msg = _classify_api_error(exc)
                print(f"[LLM] Gemini model={model} failed category={category}")
                if category == "auth":
                    raise RuntimeError(clean_msg) from exc
                categories.append(category)
                errors.append(f"{model}: {clean_msg}")
                continue
            except Exception as exc:
                category, clean_msg = _classify_api_error(exc)
                print(f"[LLM] Gemini model={model} failed category={category}")
                if category == "auth":
                    raise RuntimeError(clean_msg) from exc
                categories.append(category)
                errors.append(f"{model}: {clean_msg}")
                continue

            text = _response_text(response)
            if text:
                print(f"[LLM] Gemini model={model} succeeded response_chars={len(text)}")
                return text

            print(f"[LLM] Gemini model={model} returned empty text")
            categories.append("empty")
            errors.append(f"{model}: empty response")

        detail = "\n".join(errors) if errors else "no model was called"

        if categories and all(cat == "quota" for cat in categories):
            raise RuntimeError(
                "Gemini API quota or rate limit exceeded across available models. "
                "Wait a moment and retry, or check your Google AI Studio quota.\n"
                f"Details:\n{detail}"
            )

        if "quota" in categories:
            raise RuntimeError(
                "Gemini API quota or rate limit was hit while processing this request.\n"
                f"Details:\n{detail}"
            )

        if categories and all(cat == "model" for cat in categories):
            raise RuntimeError(
                "None of the configured Gemini models could be accessed. "
                "Check GEMINI_MODEL in .env or Streamlit secrets.\n"
                f"Details:\n{detail}"
            )

        raise RuntimeError(f"The Gemini language model request failed.\n{detail}")

    return RunnableLambda(_call)


def invoke_text(chain, payload):
    try:
        result = chain.invoke(payload)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"The Gemini language model request failed: {exc}"
        ) from exc

    if not str(result).strip():
        raise RuntimeError(
            "The Gemini language model returned an empty response. Try again in a moment."
        )

    return result
