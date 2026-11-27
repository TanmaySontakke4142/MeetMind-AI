import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Each meeting gets its own unique database directory.
CHROMA_DIR = None

_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

    return _embeddings


def build_vector_store(transcript: str) -> Chroma:
    global CHROMA_DIR

    print("Building vector store")

    text = (transcript or "").strip()

    if not text:
        raise ValueError("There is no transcript to index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    # Create a brand-new directory. Never delete or reuse an open database.
    CHROMA_DIR = tempfile.mkdtemp(prefix="meetmind_chroma_")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector database created at: {CHROMA_DIR}")

    return vector_store


def load_vector_store() -> Chroma:
    if not CHROMA_DIR or not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            "Vector database does not exist. Analyze a meeting first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
