"""
Ingest PDF documents into Qdrant with structure-aware chunking.

Two parsers:
  Descriptive (vols 1-9): all PDFs concatenated into one stream, then split at
  species headers (COMMON_NAME | Scientific name | Status).  Every sub-chunk
  from a species block carries {common_name, scientific_name, status} in its
  payload so retrieval always knows which bird a fragment is about.

  Checklist (vol 10, auto-detected by absence of pipe headers): each numbered
  row becomes one compact document with structured metadata.

Usage (from backend/ directory):
    uv run python scripts/ingest.py
"""

import bisect
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.services.embedding_service import EmbeddingService, SparseEmbeddingService

setup_logging()
logger = get_logger("ingest")

DATA_DIR = Path(settings.data_dir)
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
MIN_TEXT_LENGTH = 30

# Matches: COMMON_NAME | Scientific name [| STATUS]
# Common name: starts with uppercase letter, may include parenthetical aliases
#   e.g.  GARZA BRUJA CORONADA (Garza Encapuchada)  or  ALBATROS "TIPO CAUTA"
# Scientific name: starts with uppercase letter, may have parens (subspecies)
# STATUS is optional (absent in sections like "Posibles cambios taxonómicos")
_SPECIES_RE = re.compile(
    r"^([A-ZÁÉÍÓÚÜÑ][^|\n]+?)\s*\|\s*([A-Z][^|\n]+?)\s*(?:\|\s*([^\n]*))?$",
    re.MULTILINE,
)

# Status abbreviations used throughout the book
_STATUS_TOKENS = {"N", "V", "R", "E", "I", "Ma", "Mp", "Vi", "Mn", "CT"}

# Strips repeating page headers (book title line + optional column-header row).
# Appears at the top of every taxonomy page.
_HEADER_RE = re.compile(
    r"\d+\s*\nLISTADO DE LAS AVES ARGENTINAS\n\d+\n"
    r"TEMAS DE NATURALEZA Y CONSERVACIÓN - MONOGRAFÍA DE AVES ARGENTINAS Nº \d+\n"
    r"(?:NOMBRE CIENTÍFICO\nNOMBRE VULGAR\nNOMBRE INGLÉS\nCÓDIGO\nNRO\n)?",
    re.IGNORECASE,
)


def clean_page_text(text: str) -> str:
    return _HEADER_RE.sub("", text).strip()


def extract_page_images(
    doc: fitz.Document, page: fitz.Page, page_num: int, pdf_stem: str
) -> list[str]:
    filenames = []
    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue
        ext = base_image.get("ext", "png")
        if base_image.get("width", 0) < 80 or base_image.get("height", 0) < 80:
            continue
        filename = f"{pdf_stem}_p{page_num + 1}_{img_idx + 1}.{ext}"
        (IMAGES_DIR / filename).write_bytes(base_image["image"])
        filenames.append(filename)
    return filenames


def _is_status_line(line: str) -> bool:
    """True if line looks like a species status code (N, R, N V Mn, etc.)."""
    if not line or len(line) > 20:
        return False
    tokens = set(line.replace("(", "").replace(")", "").split())
    return bool(tokens & _STATUS_TOKENS)


# ── Page index ─────────────────────────────────────────────────────────────────

def _build_page_index(pdf_paths: list[Path]) -> tuple[str, list[int], list[tuple]]:
    """
    Concatenate clean text from all pages across all given PDFs in order.

    Returns:
      full_text   — single string of all pages
      page_starts — sorted list of char offsets where each page begins
      page_meta   — (source_filename, 1-based page_num, image_filenames) per page
    """
    full_text = ""
    page_starts: list[int] = []
    page_meta: list[tuple] = []

    for pdf_path in sorted(pdf_paths):
        logger.info("Indexing %s", pdf_path.name)
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = clean_page_text(page.get_text("text"))
            images = extract_page_images(doc, page, page_num, pdf_path.stem)
            if text:
                page_starts.append(len(full_text))
                page_meta.append((pdf_path.name, page_num + 1, images))
                full_text += text + "\n"
        doc.close()

    return full_text, page_starts, page_meta


def _page_for_offset(
    offset: int, page_starts: list[int], page_meta: list[tuple]
) -> tuple[str, int]:
    idx = max(0, bisect.bisect_right(page_starts, offset) - 1)
    source, page, _ = page_meta[idx]
    return source, page


def _images_for_span(
    start: int, end: int, page_starts: list[int], page_meta: list[tuple]
) -> list[str]:
    s_idx = max(0, bisect.bisect_right(page_starts, start) - 1)
    e_idx = bisect.bisect_left(page_starts, end)
    seen: set[str] = set()
    images: list[str] = []
    for i in range(s_idx, min(e_idx + 1, len(page_meta))):
        for fn in page_meta[i][2]:
            if fn not in seen:
                images.append(fn)
                seen.add(fn)
    return images


# ── Descriptive parser (vols 1-9) ─────────────────────────────────────────────

def parse_descriptive_pdfs(
    pdf_paths: list[Path], splitter: RecursiveCharacterTextSplitter
) -> list[dict]:
    """
    Treat all descriptive PDFs as one continuous text stream (species descriptions
    span across volume boundaries).  Split at species headers; every resulting
    sub-chunk carries {common_name, scientific_name, status} in its payload.
    """
    full_text, page_starts, page_meta = _build_page_index(pdf_paths)

    all_matches = list(_SPECIES_RE.finditer(full_text))

    # Drop the legend row that appears in the introduction ("NOMBRE COMÚN | Nombre científico")
    matches = [
        m for m in all_matches
        if m.group(1).strip().upper() != "NOMBRE COMÚN"
    ]

    logger.info("Found %d species headers across descriptive volumes", len(matches))

    chunks: list[dict] = []
    for i, match in enumerate(matches):
        common_name = match.group(1).strip()
        scientific_name = match.group(2).strip()
        status = (match.group(3) or "").strip()

        block_start = match.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[block_start:block_end].strip()

        if len(block) < MIN_TEXT_LENGTH:
            continue

        source, page = _page_for_offset(block_start, page_starts, page_meta)
        images = _images_for_span(block_start, block_end, page_starts, page_meta)

        for sub in splitter.split_text(block):
            sub = sub.strip()
            if len(sub) < MIN_TEXT_LENGTH:
                continue
            chunks.append({
                "text": sub,
                "common_name": common_name,
                "scientific_name": scientific_name,
                "status": status,
                "source": source,
                "page": page,
                "chunk_id": str(uuid.uuid4()),
                "image_filenames": images,
            })

    logger.info(
        "Descriptive volumes → %d chunks from %d species", len(chunks), len(matches)
    )
    return chunks


# ── Checklist parser (vol 10) ─────────────────────────────────────────────────

def parse_checklist_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse the numbered reference checklist (Apéndice 1).
    Format per entry: number / scientific_name / spanish_name / english_name / [status]
    Each entry → one compact document with structured metadata.
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for p in doc:
        text = clean_page_text(p.get_text("text"))
        if text:
            full_text += text + "\n"
    doc.close()

    # Split on lines that are a bare number (entry boundaries)
    number_positions = [
        (m.start(), int(m.group()))
        for m in re.finditer(r"(?m)^\d{1,4}$", full_text)
        if int(m.group()) <= 1500  # cap to valid species range; filters stray years
    ]

    chunks: list[dict] = []
    seen_numbers: set[int] = set()

    for i, (pos, number) in enumerate(number_positions):
        if number in seen_numbers:
            continue
        seen_numbers.add(number)

        end = number_positions[i + 1][0] if i + 1 < len(number_positions) else len(full_text)
        lines = [ln.strip() for ln in full_text[pos:end].split("\n") if ln.strip()]

        # Minimum: number + scientific + spanish + english
        if len(lines) < 4:
            continue

        scientific = lines[1]
        spanish = lines[2]
        english = lines[3]

        # Scientific name must follow binomial convention (Genus species...)
        if not re.match(r"^[A-Z][a-záéíóúüñ]", scientific):
            continue

        # Reject column-header artifacts
        skip_words = {"NOMBRE", "CÓDIGO", "NRO", "LISTADO", "IFORMES", "IDAE"}
        if any(w in spanish.upper() for w in skip_words):
            continue

        status = lines[4] if len(lines) > 4 and _is_status_line(lines[4]) else ""

        text = f"{number}. {spanish} ({scientific}) / {english}"
        if status:
            text += f" [{status}]"

        chunks.append({
            "text": text,
            "common_name": spanish,
            "scientific_name": scientific,
            "english_name": english,
            "status": status,
            "number": number,
            "source": pdf_path.name,
            "page": 1,
            "chunk_id": str(uuid.uuid4()),
            "image_filenames": [],
        })

    logger.info("Checklist %s → %d entries", pdf_path.name, len(chunks))
    return chunks


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def recreate_collection(client: QdrantClient, vector_size: int) -> None:
    name = settings.qdrant_collection_name
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)
        logger.info("Deleted existing collection '%s'", name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    logger.info("Created collection '%s' (dim=%d, cosine + BM25 sparse)", name, vector_size)


def upsert_batch(
    client: QdrantClient,
    chunks: list[dict],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
) -> None:
    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector={"dense": dense, "sparse": sparse},
            payload=chunk,
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection_name, points=points)


# ── Volume classification ──────────────────────────────────────────────────────

def _is_checklist(pdf_path: Path) -> bool:
    """Detect the reference checklist volume by the absence of pipe-based species headers."""
    doc = fitz.open(str(pdf_path))
    full = "".join(p.get_text("text") for p in doc)
    doc.close()
    return "|" not in full


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s — add PDFs and re-run.", DATA_DIR)
        return

    descriptive_pdfs = [p for p in pdf_files if not _is_checklist(p)]
    checklist_pdfs = [p for p in pdf_files if _is_checklist(p)]
    logger.info(
        "Classified: %d descriptive volume(s), %d checklist volume(s)",
        len(descriptive_pdfs),
        len(checklist_pdfs),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    embed_svc = EmbeddingService()
    sparse_embed_svc = SparseEmbeddingService()
    client = QdrantClient(url=settings.qdrant_url)

    vector_size = len(embed_svc.embed_one("probe"))
    logger.info("Detected vector size: %d", vector_size)
    recreate_collection(client, vector_size)

    all_chunks: list[dict] = []
    if descriptive_pdfs:
        all_chunks.extend(parse_descriptive_pdfs(descriptive_pdfs, splitter))
    for pdf_path in checklist_pdfs:
        all_chunks.extend(parse_checklist_pdf(pdf_path))

    logger.info("Total chunks to embed: %d", len(all_chunks))

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        dense_vectors = embed_svc.embed_batch(texts)
        sparse_vectors = sparse_embed_svc.embed_batch(texts)
        upsert_batch(client, batch, dense_vectors, sparse_vectors)
        logger.info("Upserted %d/%d", min(i + BATCH_SIZE, len(all_chunks)), len(all_chunks))

    logger.info("Ingestion complete. %d chunks stored in Qdrant.", len(all_chunks))


if __name__ == "__main__":
    main()
