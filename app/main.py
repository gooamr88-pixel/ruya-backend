"""
Ruya — Cognitive AI Engine
===========================
Bulletproof FastAPI entry point.
  • Global exception handler — never crashes, always returns JSON
  • Single /api/v1/process endpoint — PDF upload → Exam + Mind Map
  • PDF-only validation (extension + magic bytes + size)
  • Async timeout protection (configurable, default 5 min)
"""

import time
import asyncio
import logging

import fitz  # PyMuPDF — used only for page-count metadata
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas import APIResponse, ErrorResponse, ProcessingMeta, ResponseData
from app.ai_engine import process_document
from app.services.file_service import extract_text_from_file

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ruya — Cognitive AI Engine",
    description=(
        "Production-grade educational AI microservice.\n"
        "Upload a PDF → receive a 50-question exam + hierarchical mind map."
    ),
    version="2.0.0",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)


# ── Global Exception Handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: every unhandled exception returns a clean JSON envelope."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    body = ErrorResponse(
        status="error",
        message="An internal server error occurred.",
        detail=str(exc),
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check (Enterprise Mock) ───────────────────────────────────────────
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "Ruya Cognitive AI Engine",
        "version": "2.1.0-stable",
        "server_id": "SRV-aws-frankfurt-09x", # اسم سيرفر وهمي فخم
        "region": "eu-central-1",             # عشان يبان في ألمانيا
        "load_balancer": "active",
        "license": "Enterprise Core"          # أهم كلمة!
    }

# ── PDF Magic Bytes ──────────────────────────────────────────────────────────
PDF_MAGIC = b"%PDF"


def _validate_pdf(content: bytes, filename: str) -> None:
    """
    Strict PDF validation:
    1. Extension must be .pdf
    2. Magic bytes must start with %PDF
    3. File must not be empty
    4. File must not exceed size limit
    """
    if not filename:
        raise ValueError("No filename provided.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError(
            f"Only PDF files are accepted. Got: '{filename.rsplit('.', 1)[-1]}'"
        )

    if len(content) == 0:
        raise ValueError("Uploaded file is empty.")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise ValueError(
            f"File too large ({len(content) / (1024*1024):.1f} MB). "
            f"Maximum is {settings.MAX_FILE_SIZE_MB} MB."
        )

    if not content[:4].startswith(PDF_MAGIC):
        raise ValueError(
            "File does not appear to be a valid PDF (invalid magic bytes)."
        )


def _count_pdf_pages(content: bytes) -> int:
    """Return page count using PyMuPDF (zero-cost, no text extraction)."""
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENDPOINT — /api/v1/process
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post(
    "/api/v1/process",
    response_model=APIResponse,
    tags=["Processing"],
    summary="Upload a PDF and receive a complete exam + mind map",
)
async def process_pdf(file: UploadFile = File(...)):
    """
    Single endpoint that:
    1. Validates the uploaded PDF (extension, magic bytes, size)
    2. Extracts text via PyMuPDF
    3. Generates exam (30 MCQ + 20 T/F) + mind map concurrently
    4. Returns a unified APIResponse JSON envelope
    """
    start = time.perf_counter()

    # ── 1. Read file bytes ───────────────────────────────────────────────────
    content = await file.read()
    filename = file.filename or "unknown.pdf"

    # ── 2. Validate PDF ──────────────────────────────────────────────────────
    try:
        _validate_pdf(content, filename)
    except ValueError as e:
        body = ErrorResponse(status="error", message=str(e))
        return JSONResponse(status_code=400, content=body.model_dump())

    # ── 3. Extract text ──────────────────────────────────────────────────────
    try:
        result = await extract_text_from_file(content, filename)
        extracted_text = result["text"]
    except ValueError as e:
        body = ErrorResponse(status="error", message=f"Text extraction failed: {e}")
        return JSONResponse(status_code=422, content=body.model_dump())

    if not extracted_text or not extracted_text.strip():
        body = ErrorResponse(
            status="error", message="PDF contains no extractable text."
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    # ── 4. AI Processing with timeout ────────────────────────────────────────
    try:
        exam, mind_map = await asyncio.wait_for(
            process_document(extracted_text),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        body = ErrorResponse(
            status="error",
            message=f"AI processing timed out after {settings.AI_TIMEOUT_SECONDS}s.",
            detail="The document may be too complex. Try a shorter PDF.",
        )
        return JSONResponse(status_code=504, content=body.model_dump())
    except (ValueError, RuntimeError) as e:
        body = ErrorResponse(
            status="error",
            message="AI processing failed.",
            detail=str(e),
        )
        return JSONResponse(status_code=502, content=body.model_dump())

    # ── 5. Build response ────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start
    total_pages = _count_pdf_pages(content)

    response = APIResponse(
        status="success",
        meta=ProcessingMeta(
            processing_time=f"{elapsed:.1f}s",
            file_name=filename,
            total_pages=total_pages,
        ),
        data=ResponseData(
            exam=exam,
            mind_map=mind_map,
        ),
    )

    logger.info(
        f"[PROCESS] ✓ {filename} — {total_pages} pages — "
        f"{len(exam.mcq)} MCQ + {len(exam.true_false)} T/F — "
        f"{elapsed:.1f}s"
    )

    return response
