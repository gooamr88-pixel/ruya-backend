"""
Ruya — Unified Pydantic Schemas
================================
Strict C#-interoperable JSON contract.
Every response from this API is wrapped in APIResponse or ErrorResponse.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Exam Models ──────────────────────────────────────────────────────────────

class MCQQuestion(BaseModel):
    """A multiple-choice question with exactly 4 options (A/B/C/D)."""
    id: int
    question: str
    options: dict  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str = Field(..., pattern=r"^[A-D]$", description="Correct option letter")


class TrueFalseQuestion(BaseModel):
    """A True/False question."""
    id: int
    question: str
    answer: bool


class ExamData(BaseModel):
    """Complete exam payload: 30 MCQ + 20 True/False = 50 questions."""
    mcq: List[MCQQuestion]
    true_false: List[TrueFalseQuestion]


# ── Mind Map Models ──────────────────────────────────────────────────────────

class MindMapNode(BaseModel):
    """Recursive tree node for hierarchical mind maps."""
    id: str
    label: str
    children: Optional[List[MindMapNode]] = []


# ── Processing Metadata ─────────────────────────────────────────────────────

class ProcessingMeta(BaseModel):
    """Metadata about the processing run."""
    processing_time: str = Field(..., description="e.g. '12.4s'")
    file_name: str
    total_pages: int


# ── Unified Response ─────────────────────────────────────────────────────────

class ResponseData(BaseModel):
    """Top-level data payload containing exam + mind map."""
    exam: ExamData
    mind_map: MindMapNode


class APIResponse(BaseModel):
    """
    Standard success envelope.
    C# client always deserialises this shape.
    """
    status: str = "success"
    meta: ProcessingMeta
    data: ResponseData


class ErrorResponse(BaseModel):
    """
    Standard error envelope.
    C# client always deserialises this shape.
    """
    status: str = "error"
    message: str
    detail: Optional[str] = None
