from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm, invoke_text


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    text = (transcript or "").strip()
    if not text:
        raise ValueError("There is no transcript to summarise.")

    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript in a few concise bullet points."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(text)
    print(f"[Summarizer] Summarizing {len(chunks)} chunk(s) (transcript_chars={len(text)})")
    chunk_summaries = [
        invoke_text(map_chain, {"text": chunk})
        for chunk in chunks
    ]
    if len(chunk_summaries) == 1:
        print(f"[Summarizer] Summary generated (chars={len(chunk_summaries[0])})")
        return chunk_summaries[0]

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one professional meeting summary in bullet points. Do not invent facts.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = (
        RunnableLambda(lambda raw: {"text": raw})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    combined = invoke_text(combined_chain, "\n\n".join(chunk_summaries))
    print(f"[Summarizer] Combined summary generated (chars={len(combined)})")
    return combined


def generate_title(transcript: str) -> str:
    text = (transcript or "").strip()
    if not text:
        return "Meeting brief"

    print(f"[Summarizer] Generating title (input_chars={min(len(text), 2000)})")
    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Write a short meeting title of at most 8 words. "
            "Return only the title, with no quotes.",
        ),
        ("human", "{text}"),
    ])
    title_chain = (
        RunnableLambda(lambda raw: {"text": raw})
        | title_prompt
        | get_llm(max_tokens=200)
        | StrOutputParser()
    )
    raw = str(invoke_text(title_chain, text[:2000])).strip().strip('"')
    title = raw.splitlines()[0].strip() if raw else ""
    print(f"[Summarizer] Title generated (chars={len(title)})")
    return title or "Meeting brief"
