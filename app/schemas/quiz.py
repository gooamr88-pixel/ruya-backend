from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


# ── Request ──────────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    """Request body for quiz generation."""
    text: str = Field(..., min_length=20, description="Educational text to generate quiz from")
    num_questions: int = Field(default=5, ge=1, le=30, description="Number of questions to generate")
    difficulty: Difficulty = Field(default=Difficulty.medium, description="Desired difficulty level")


# ── Response ─────────────────────────────────────────────────────────────────

class QuizOption(BaseModel):
    """A single answer option."""
    text: str
    is_correct: bool


class QuizQuestion(BaseModel):
    """A single quiz question with options and explanation."""
    question: str
    options: List[QuizOption]
    explanation: str
    difficulty: str


class QuizResponse(BaseModel):
    """Full quiz response returned to the client."""
    title: str
    questions: List[QuizQuestion]
