from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
import fitz  # PyMuPDF
import asyncio
import os
from dotenv import load_dotenv

# ── 1. تعريف الـ Schemas (كانت في ملف لوحدها وجبناها هنا) ──────────────
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

# ── 2. إعداد التطبيق ───────────────────────────────────────────────────
load_dotenv()

app = FastAPI(
    title="Ruya Core Engine",
    version="3.0.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 3. دوال مساعدة (استخراج النص) ──────────────────────────────────────
def extract_text_from_pdf(content: bytes) -> str:
    try:
        text = ""
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# ── 4. الـ Endpoint الرئيسية (API) ─────────────────────────────────────
@app.post("/api/v1/process", response_model=APIResponse, tags=["Core"])
async def process_document(file: UploadFile = File(...)):
    # 1. قراءة الملف
    content = await file.read()
    filename = file.filename
    
    # 2. استخراج النص (محاكاة عشان السرعة دلوقتي)
    text = extract_text_from_pdf(content)
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty or invalid PDF")

    # 3. تجهيز رد (Mock Response) عشان نتأكد إن السيرفر شغال
    # (بعد ما يشتغل هنحط كود الـ AI هنا)
    
    mock_exam = Exam(
        mcq=[
            Question(id=1, text="Is Ruya API working?", options=["Yes", "No"], correct_answer="Yes", type="MCQ"),
            Question(id=2, text="Is Vercel fast?", options=["Yes", "Maybe"], correct_answer="Yes", type="MCQ")
        ],
        true_false=[
            Question(id=1, text="Python is awesome", options=None, correct_answer="True", type="TrueFalse")
        ]
    )
    
    # 4. الرد النهائي
    return APIResponse(
        status="success",
        meta=ProcessingMeta(
            processing_time="0.05s",
            file_name=filename,
            total_pages=1
        ),
        data=ResponseData(
            exam=mock_exam,
            mind_map={"root": "Ruya Project", "children": [{"id": "1", "label": "Working!"}]}
        )
    )

# ── Health Check ─────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def health_check():
    return {"status": "operational", "server": "Vercel Edge"}