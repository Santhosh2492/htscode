"""Build the local Chroma RAG index from tariff.pdf without langchain-community."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "tariff.pdf"
DEFAULT_DB = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = os.getenv("HTS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Tariff PDF not found: {pdf_path}. Place the official tariff PDF in the project root."
        )
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": str(pdf_path.name), "page": page_number},
            ))
    if not docs:
        raise ValueError("No extractable text was found in the PDF. If it is scanned, OCR is needed.")
    return docs


def ingest_data(pdf_path: str | Path = DEFAULT_PDF, persist_directory: str | Path = DEFAULT_DB,
                reset: bool = False) -> Chroma:
    pdf_path = Path(pdf_path).expanduser().resolve()
    persist_directory = Path(persist_directory).expanduser().resolve()
    if reset and persist_directory.exists():
        import shutil
        shutil.rmtree(persist_directory)

    print(f"Reading {pdf_path} ...")
    documents = load_pdf_documents(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=180)
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages; created {len(chunks)} text chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name="canadian_tariff",
    )
    print(f"Index saved to {persist_directory}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the local Canadian tariff RAG index.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to the official tariff PDF")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Chroma database directory")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the existing index")
    args = parser.parse_args()
    ingest_data(args.pdf, args.db, args.reset)
