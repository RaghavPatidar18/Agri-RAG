import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.console import Console

from document_reader import process_pdf, process_txt, process_pptx

load_dotenv()

console = Console()

FAISS_INDEX_PATH = "./ocr_index"
DOCS_FOLDER = "./documents"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def text_to_documents(text: str, filename: str, file_type: str) -> list[Document]:
    """
    Converts raw OCR-extracted text into Document objects with metadata.
    Splits PDF text by page markers and PPTX text by slide markers so that
    each section gets proper metadata (source, file_type, page/slide number).
    """
    docs = []

    if file_type == "pdf":
        # process_pdf() produces: "--- Page 1 ---\ntext\n--- Page 2 ---\ntext"
        # re.split with a capturing group gives: ['', '1', 'text1', '2', 'text2', ...]
        parts = re.split(r'--- Page (\d+) ---', text)
        for i in range(1, len(parts), 2):
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "file_type": file_type,
                        "page": int(parts[i]),
                    }
                ))

    elif file_type == "pptx":
        # process_pptx() produces: "--- Slide 1 ---\ntext\n\n--- Slide 2 ---\ntext"
        parts = re.split(r'--- Slide (\d+) ---', text)
        for i in range(1, len(parts), 2):
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "file_type": file_type,
                        "slide": int(parts[i]),
                    }
                ))

    else:  # txt — plain text, no markers
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "file_type": file_type}
            ))

    # Fallback: if parsing produced nothing, store the whole text as one document
    if not docs and text.strip():
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "file_type": file_type}
        ))

    return docs


def index_all_documents() -> int:
    """
    Scans DOCS_FOLDER, runs EasyOCR on every supported file (PDF, TXT, PPTX),
    chunks the extracted text, embeds it with HuggingFace, and persists to a
    local FAISS index at FAISS_INDEX_PATH.

    Returns the total number of chunks indexed.
    """
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)

    all_docs: list[Document] = []

    for filename in sorted(os.listdir(DOCS_FOLDER)):
        path = os.path.join(DOCS_FOLDER, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                console.print(f"\n[bold cyan]Processing PDF:[/] {filename}")
                text = process_pdf(path)
                docs = text_to_documents(text, filename, "pdf")

            elif ext == ".txt":
                console.print(f"\n[bold cyan]Processing TXT:[/] {filename}")
                text = process_txt(path)
                docs = text_to_documents(text, filename, "txt")

            elif ext == ".pptx":
                console.print(f"\n[bold cyan]Processing PPTX:[/] {filename}")
                text = process_pptx(path)
                docs = text_to_documents(text, filename, "pptx")

            else:
                continue

            all_docs.extend(docs)
            console.print(f"[bold green]✓[/] Extracted {len(docs)} section(s) from {filename}")

        except Exception as e:
            console.print(f"[bold red]Error processing {filename}:[/] {e}")
            continue

    if not all_docs:
        console.print("[bold yellow]No documents found to index.[/]")
        return 0

    console.print(f"\n[bold]Chunking {len(all_docs)} section(s) into smaller pieces...[/]")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    console.print(f"[bold]Embedding {len(chunks)} chunks and building FAISS index...[/]")
    with console.status("[bold green]Saving FAISS index..."):
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

    console.print(f"\n[bold green]✓ Done![/] {len(chunks)} chunks saved to [cyan]{FAISS_INDEX_PATH}[/]")
    return len(chunks)


def get_retriever(k: int = 4):
    """Loads the OCR FAISS index and returns a LangChain retriever, or None if not built yet."""
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={"k": k})
    return None


if __name__ == "__main__":
    total = index_all_documents()
    console.print(f"\n[bold]Total chunks indexed:[/] {total}")
