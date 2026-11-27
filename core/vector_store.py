import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use a writable temporary location instead of the repository directory.
ROOT = Path(tempfile.gettempdir()) / "meetmind_ai"
CHROMA_DIR = str(ROOT / "vector_db")

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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

    # Create the database in a writable temporary directory.
    ROOT.mkdir(parents=True, exist_ok=True)

    # Remove the previous database before creating a new meeting database.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print("Vector store created successfully.")

    return vector_store


def load_vector_store() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
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
