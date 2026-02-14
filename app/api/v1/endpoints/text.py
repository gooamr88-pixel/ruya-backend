import json
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.openai_service import (
    generate_quiz,
    generate_question_bank,
    generate_mindmap,
    generate_quiz_stream,
    generate_question_bank_stream,
    generate_mindmap_stream,
)
from app.services.file_service import extract_text_from_file
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    text: str = Field(..., min_length=20)
    num_questions: int = Field(default=5, ge=1, le=30)
    difficulty: str = Field(default="medium")

class MindMapRequest(BaseModel):
    text: str = Field(..., min_length=20)


# ── Helper: SSE Event Stream ─────────────────────────────────────────────────

async def _sse_wrapper(generator):
    """Wraps an async generator into SSE format."""
    try:
        async for chunk in generator:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. QUIZ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/quiz")
async def create_quiz(request: QuizRequest):
    """Generate a quiz from educational text."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        result = await generate_quiz(request.text, request.num_questions, request.difficulty)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/text/quiz/stream")
async def create_quiz_stream(request: QuizRequest):
    """Stream quiz generation via Server-Sent Events."""
    return StreamingResponse(
        _sse_wrapper(generate_quiz_stream(request.text, request.num_questions, request.difficulty)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. QUESTION BANK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/question-bank")
async def create_question_bank(request: QuizRequest):
    """Generate a large question bank from educational text."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        result = await generate_question_bank(request.text, request.num_questions, request.difficulty)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Question bank generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/text/question-bank/stream")
async def create_question_bank_stream(request: QuizRequest):
    """Stream question bank generation via Server-Sent Events."""
    return StreamingResponse(
        _sse_wrapper(generate_question_bank_stream(request.text, request.num_questions, request.difficulty)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MIND MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/mindmap")
async def create_mindmap(request: MindMapRequest):
    """Generate a hierarchical mind map from text."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        result = await generate_mindmap(request.text)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Mind map generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/text/mindmap/stream")
async def create_mindmap_stream(request: MindMapRequest):
    """Stream mind map generation via Server-Sent Events."""
    return StreamingResponse(
        _sse_wrapper(generate_mindmap_stream(request.text)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. FILE UPLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF/Image) and extract text from it."""
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        content = await file.read()

        # Validate size (server-side)
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await extract_text_from_file(content, file.filename)

        if not result.get("success"):
            raise HTTPException(status_code=422, detail="Text extraction failed.")

        return {
            "text": result["text"],
            "success": True,
        }

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")