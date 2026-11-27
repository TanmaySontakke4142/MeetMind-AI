"""Shared OpenRouter client. The key is read from the environment, never hardcoded."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openrouter import ChatOpenRouter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Preferred model
DEFAULT_MODEL = "baidu/ernie-5.5-flash-preview:free"

FALLBACK_MODELS = (
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "liquid/lfm-3-1.7b:free",
    "openrouter/free",
)

_clients: dict[tuple[str, int], ChatOpenRouter] = {}


def require_api_key() -> str:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to the .env file in the project folder."
        )
    return key


def _model_order() -> list[str]:
    preferred = (os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    ordered = [preferred, *FALLBACK_MODELS]
    seen = set()
    unique = []

    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)

    return unique


def _client(model: str, max_tokens: int) -> ChatOpenRouter:
    key = (model, max_tokens)

    if key not in _clients:
        _clients[key] = ChatOpenRouter(
            model=model,
            api_key=require_api_key(),
            temperature=0,
            max_tokens=max_tokens,
            app_title="MeetMind AI",
        )

    return _clients[key]


def _message_text(message) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))

        content = "".join(parts)

    return str(content or "").strip()


def get_llm(max_tokens: int = 1200):
    """Runnable that tries the preferred free model, then other free models."""

    def _call(value):
        messages = value.to_messages() if hasattr(value, "to_messages") else value
        errors = []

        for model in _model_order():
            try:
                message = _client(model, max_tokens).invoke(messages)
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                continue

            text = _message_text(message)

            if text:
                return text

            errors.append(f"{model}: empty response")

        # Display all attempted model errors
        detail = "\n".join(errors) if errors else "no model was called"

        lowered = detail.lower()

        if any(token in lowered for token in ("429", "rate", "too many")):
            raise RuntimeError(
                "OpenRouter is rate-limiting the free models. Wait a minute and try again. "
                f"Last error: {detail}"
            )

        raise RuntimeError(
            f"The language model request failed. {detail}"
        )

    return RunnableLambda(_call)


def invoke_text(chain, payload):
    try:
        result = chain.invoke(payload)

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"The language model request failed: {exc}"
        ) from exc

    if not str(result).strip():
        raise RuntimeError(
            "The language model returned an empty response. Try again in a moment."
        )

    return result