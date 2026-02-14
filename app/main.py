"""
Ruya — Cognitive AI Engine (Serverless Edition)
===============================================
Production-Ready Main Entry Point.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
import fitz  # PyMuPDF
import asyncio
import logging
import time
from dotenv import load_dotenv

# استدعاء دالة الـ AI (تأكد إن ملف ai_engine.py مفيهوش imports بايظة)
from app.ai_engine import process_document

# ── 1. إعدادات الـ Logging (عشان نعرف لو حصل خطأ) ──────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RuyaCore")

# ── 2. تعريف الـ Schemas (داخلياً لمنع مشاكل الاستيراد) ────────────────
class Question(BaseModel):
    id: int
    text: str
    options: Optional[List[str]] = None
    correct_answer: str
    type: str

class Exam(BaseModel):
    mcq: List[Question]
    true_false: List[Question]

class MindMapNode(BaseModel):
    id: str
    label: str
    children: List['MindMapNode'] = []

MindMapNode.model_rebuild()

class ProcessingMeta(BaseModel):
    processing_time: str
    file_name: str
    total_pages: int

class ResponseData(BaseModel):
    exam: Exam
    mind_map: Dict[str, Any]

class APIResponse(BaseModel):
    status: str
    meta: ProcessingMeta
    data: ResponseData

class ErrorResponse(BaseModel):
    status: str
    message: str
    detail: Optional[str] = None

# ── 3. إعداد التطبيق ───────────────────────────────────────────────────
load_dotenv()

app = FastAPI(
    title="Ruya Core Engine",
    description="Scalable AI Backend for Ruya Platform. Powered by Vercel Edge.",
    version="3.0.0-production",
    docs_url="/docs",
    redoc_url=None
)

# تفعيل الـ CORS عشان الويب والموبايل يعرفوا يكلموه
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handler (عشان السيرفر مايقعش أبداً) ───────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal Server Error", "detail": str(exc)}
    )

# ── 4. دوال مساعدة ─────────────────────────────────────────────────────
def extract_text_from_pdf(content: bytes) -> str:
    """استخراج النص من ملف PDF باستخدام PyMuPDF"""
    try:
        text = ""
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        raise ValueError("Failed to extract text from PDF")

# ── 5. الـ Endpoint الرئيسية (الذكاء الاصطناعي الحقيقي) ───────────────
@app.post("/api/v1/process", response_model=APIResponse, tags=["Core"])
async def process_pdf_endpoint(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    filename = file.filename or "unknown.pdf"

    # 1. قراءة الملف
    content = await file.read()
    
    # 2. التحقق البسيط (Validation)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # 3. استخراج النص
    try:
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or scanned (no text found).")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4. المعالجة بالذكاء الاصطناعي (Real AI)
    try:
        # بنعمل Timeout عشان لو طول يفصل (120 ثانية مثلاً)
        # لاحظ: فيرسيل المجاني بيفصل بعد 10-60 ثانية حسب الضغط
        exam_data, mind_map_data = await asyncio.wait_for(
            process_document(text),
            timeout=50.0 
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI processing timed out (Document too large).")
    except Exception as e:
        logger.error(f"AI Engine Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

    # 5. حساب الوقت وبناء الرد
    elapsed = f"{time.perf_counter() - start_time:.1f}s"
    
    return APIResponse(
        status="success",
        meta=ProcessingMeta(
            processing_time=elapsed,
            file_name=filename,
            total_pages=0 # مش مهم نحسب الصفحات دلوقتي عشان السرعة
        ),
        data=ResponseData(
            exam=exam_data,
            mind_map=mind_map_data
        )
    )

# ── Health Check (Enterprise Mock) ───────────────────────────────────
@app.get("/", tags=["System"])
def health_check():
    return {
        "status": "operational",
        "service": "Ruya Cognitive AI Engine",
        "version": "3.0.0-stable",
        "region": "eu-central-1",
        "license": "Enterprise Core"
    }