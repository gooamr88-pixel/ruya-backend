import time
import asyncio
import logging
import os
import fitz  # PyMuPDF
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# استيراد من ai_engine (تأكد إن المسار صح)
from app.ai_engine import process_document, ExamData, MindMapNode

# ── 1. القوالب (Schemas) ──────────────────────────────────────────
class ProcessingMeta(BaseModel):
    processing_time: str
    file_name: str
    total_pages: int

class ResponseData(BaseModel):
    exam: ExamData
    mind_map: MindMapNode

class APIResponse(BaseModel):
    status: str
    meta: ProcessingMeta
    data: ResponseData

class ErrorResponse(BaseModel):
    status: str
    message: str
    detail: Optional[str] = None

# ── 2. إعداد التطبيق ──────────────────────────────────────────────
app = FastAPI(
    title="Ruya Enterprise Core API",
    description="Scalable AI Backend for Ruya Platform.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 3. دالة استخراج النص (نقلناها هنا بدل الملف الخارجي) ──────────────
async def extract_text_from_file(content: bytes, filename: str) -> dict:
    try:
        text = ""
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return {"text": text}
    except Exception as e:
        raise ValueError(f"PDF parsing error: {e}")

# ── 4. الـ Health Check الفخم ─────────────────────────────────────
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "Ruya Cognitive AI Engine",
        "license": "Enterprise Core"
    }

# ── 5. الـ Endpoint الرئيسية ─────────────────────────────────────
@app.post("/api/v1/process", response_model=APIResponse)
async def process_pdf(file: UploadFile = File(...)):
    start = time.perf_counter()
    content = await file.read()
    filename = file.filename or "unknown.pdf"

    # أقصى حجم ملف (مثلاً 10 ميجا) لو مش عايز تعتمد على Config خارجي
    if len(content) > 10 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"status":"error", "message":"File too large"})

    try:
        # استخراج النص
        result = await extract_text_from_file(content, filename)
        extracted_text = result["text"]

        if not extracted_text.strip():
            return JSONResponse(status_code=422, content={"status":"error", "message":"No text found"})

        # المعالجة (AI) - تايم آوت 60 ثانية لـ Vercel
        exam, mind_map = await asyncio.wait_for(
            process_document(extracted_text),
            timeout=60.0
        )

        elapsed = f"{time.perf_counter() - start:.1f}s"
        
        return APIResponse(
            status="success",
            meta=ProcessingMeta(
                processing_time=elapsed,
                file_name=filename,
                total_pages=0
            ),
            data=ResponseData(exam=exam, mind_map=mind_map)
        )

    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"status":"error", "message":"AI Timeout"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error", "message": str(e)})