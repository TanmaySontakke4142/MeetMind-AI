from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.llm import get_llm, invoke_text

# Free-model context windows vary. Long transcripts are extracted in overlapping slices.
_CHUNK_CHARS = 12000


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return (
        RunnableLambda(lambda text: {"text": text})
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def _extract(system_prompt: str, transcript: str, label: str = "items") -> str:
    text = (transcript or "").strip()
    if not text:
        raise ValueError("There is no transcript to analyse.")

    chain = build_chain(system_prompt)
    if len(text) <= _CHUNK_CHARS:
        print(f"[Extractor] Extracting {label} in 1 slice (chars={len(text)})")
        output = invoke_text(chain, text)
        print(f"[Extractor] Extracted {label} (output_chars={len(output)})")
        return output

    parts = []
    step = _CHUNK_CHARS - 500
    for start in range(0, len(text), step):
        parts.append(invoke_text(chain, text[start:start + _CHUNK_CHARS]))
    print(f"[Extractor] Merging {len(parts)} slices for {label}")
    merger = build_chain(
        "Combine these notes from different parts of one meeting into a single numbered list. "
        "Remove duplicates. Keep the requested format. "
        "If every part says nothing was found, repeat that sentence and nothing else."
    )
    output = invoke_text(merger, "\n\n".join(parts))
    print(f"[Extractor] Extracted {label} (output_chars={len(output)})")
    return output


def extract_action_items(transcript: str) -> str:
    return _extract(
        "You are an expert meeting analyst. Extract every action item, task, or commitment. "
        "For each item include the task, the assigned person when one is named, and the deadline "
        "when one is mentioned. If the person or deadline is not mentioned, write 'Not specified'.\n"
        "Format each item as:\n"
        "1. Task: ... | Assigned to: ... | Deadline: ...\n"
        "If none are found, reply exactly: No action items found.",
        transcript,
        label="action_items",
    )


def extract_key_decisions(transcript: str) -> str:
    return _extract(
        "You are an expert meeting analyst. Extract the key decisions the group agreed on. "
        "Format as a numbered list. If none are found, reply exactly: No key decisions found.",
        transcript,
        label="key_decisions",
    )


def extract_discussion_points(transcript: str) -> str:
    return _extract(
        "You are an expert meeting analyst. Extract the main discussion points. "
        "For each point give a short heading and one sentence on what was discussed. "
        "Format as a numbered list. If none are found, reply exactly: No discussion points found.",
        transcript,
        label="discussion_points",
    )


def extract_questions(transcript: str) -> str:
    return _extract(
        "From the meeting transcript, extract unresolved questions or topics that still need "
        "follow-up. Format as a numbered list. If none are found, reply exactly: No open questions found.",
        transcript,
        label="open_questions",
    )
