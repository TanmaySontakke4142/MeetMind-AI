from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.llm import get_llm
from core.vector_store import build_vector_store, get_retriever, load_vector_store

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert meeting assistant. Answer the user's question using ONLY "
        "the meeting transcript context below. Use the conversation history to understand "
        "follow-up questions, but do not add facts that are not in the context.\n\n"
        "If the answer is not in the context, say exactly: "
        "I could not find this information in the meeting transcript.\n\n"
        "Be concise. When you quote the meeting, make that clear.\n\n"
        "Conversation so far:\n{history}\n\n"
        "Context from the meeting transcript:\n{context}",
    ),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    if not docs:
        return "(no matching transcript excerpts)"
    return "\n\n".join(doc.page_content for doc in docs)


def _make_chain(retriever):
    llm = get_llm()

    def _inputs(payload: dict) -> dict:
        question = payload["question"]
        docs = retriever.invoke(question)
        return {
            "context": _format_docs(docs),
            "question": question,
            "history": payload.get("history") or "None",
        }

    return RunnableLambda(_inputs) | _PROMPT | llm | StrOutputParser()


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    chain = _make_chain(retriever)
    print("[RAG] RAG chain initialized")
    return chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    return _make_chain(retriever)


def ask_question(rag_chain, question: str, history: list | None = None) -> str:
    prior = "None"
    history_turns = len(history) if history else 0
    if history:
        prior = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in history[-6:]
        )
    print(f"[RAG] Answering question (question_chars={len(question)}, history_turns={history_turns})")
    answer = rag_chain.invoke({"question": question, "history": prior})
    print(f"[RAG] Answer generated (chars={len(str(answer))})")
    return answer
