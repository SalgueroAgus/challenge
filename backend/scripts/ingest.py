"""
Ingest PDF documents into Qdrant.

Usage (from backend/ directory):
    uv run python scripts/ingest.py

Drops all PDFs from ../data/*.pdf, extracts text + images,
chunks, embeds with FastEmbed, and stores in Qdrant.
"""

import sys
import uuid
from pathlib import Path

# Allow importing app modules when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.services.embedding_service import EmbeddingService

setup_logging()
logger = get_logger("ingest")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(settings.data_dir)
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimensions
BATCH_SIZE = 32  # chunks per Qdrant upsert call
MIN_TEXT_LENGTH = 30  # skip chunks shorter than this (noise)


def extract_page_images(
    doc: fitz.Document, page: fitz.Page, page_num: int, pdf_stem: str
) -> list[str]:
    """Extract all images from a PDF page, save to IMAGES_DIR, return filenames."""
    filenames = []
    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        ext = base_image.get("ext", "png")
        # Skip tiny images (icons, decorations) — likely not bird photos
        if base_image.get("width", 0) < 80 or base_image.get("height", 0) < 80:
            continue

        filename = f"{pdf_stem}_p{page_num + 1}_{img_idx + 1}.{ext}"
        img_path = IMAGES_DIR / filename
        img_path.write_bytes(base_image["image"])
        filenames.append(filename)

    return filenames


def process_pdf(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """
    Returns a list of chunk dicts:
      {text, source, page, chunk_id, image_filenames}
    """
    logger.info("Processing %s", pdf_path.name)
    doc = fitz.open(str(pdf_path))
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        image_filenames = extract_page_images(doc, page, page_num, pdf_path.stem)

        if not text:
            logger.info("  Page %d — no text, skipping", page_num + 1)
            continue

        page_chunks = splitter.split_text(text)
        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < MIN_TEXT_LENGTH:
                continue
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "chunk_id": str(uuid.uuid4()),
                    # All images on this page are associated with all chunks on it.
                    # This keeps retrieval simple while still surfacing relevant images.
                    "image_filenames": image_filenames,
                }
            )

    doc.close()
    image_count = sum(len(c["image_filenames"]) for c in chunks)
    logger.info("  → %d chunks, %d images", len(chunks), image_count)
    return chunks


def recreate_collection(client: QdrantClient) -> None:
    """Drop and recreate the Qdrant collection (clean re-ingest)."""
    name = settings.qdrant_collection_name
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)
        logger.info("Deleted existing collection '%s'", name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s' (dim=%d, cosine)", name, VECTOR_SIZE)


def upsert_batch(client: QdrantClient, chunks: list[dict], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "image_filenames": chunk["image_filenames"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection_name, points=points)


def main() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s — add PDFs and re-run.", DATA_DIR)
        return

    logger.info("Found %d PDF(s) in %s", len(pdf_files), DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    embed_svc = EmbeddingService()
    client = QdrantClient(url=settings.qdrant_url)

    recreate_collection(client)

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        all_chunks.extend(process_pdf(pdf_path, splitter))

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # Embed and upsert in batches
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed_svc.embed_batch(texts)
        upsert_batch(client, batch, vectors)
        upserted = min(i + BATCH_SIZE, len(all_chunks))
        logger.info("Upserted batch %d/%d", upserted, len(all_chunks))

    logger.info("Ingestion complete. %d chunks stored in Qdrant.", len(all_chunks))


if __name__ == "__main__":
    main()
