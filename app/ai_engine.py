"""
Ruya — AI Engine
=================
Handles all interactions with AI providers (Groq + Gemini) for:
  1. Exam generation  (30 MCQ + 20 True/False)
  2. Mind map generation (deep hierarchical tree)

Features:
  - Robust JSON extraction with retry logic (2 retries)
  - Multi-provider hybrid call with automatic failover
  - Text chunking for large PDFs
  - Anti-hallucination system prompts
"""

import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any

import google.generativeai as genai
from groq import AsyncGroq

from app.core.config import settings
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# ── 1. القوالب (Schemas) محلياً لمنع مشاكل الاستيراد ──────────────────
class Question(BaseModel):
    id: int
    question: str # تأكد إن الاسم 'question' مطابق للـ prompt بتاعك
    options: Dict[str, str]
    answer: str

class TrueFalseQuestion(BaseModel):
    id: int
    question: str
    answer: bool

class ExamData(BaseModel):
    mcq: List[Question]
    true_false: List[TrueFalseQuestion]

class MindMapNode(BaseModel):
    id: str
    label: str
    children: List['MindMapNode'] = []

MindMapNode.model_rebuild()

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIENT INITIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logger.info(f"[AI‑ENGINE] Provider mode: {settings.AI_PROVIDER}")

groq_client: Optional[AsyncGroq] = None
if settings.GROQ_API_KEY:
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("[AI‑ENGINE] ✓ Groq client ready")
else:
    logger.warning("[AI‑ENGINE] ✗ Groq API key missing")

if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY, transport="rest")
    logger.info("[AI‑ENGINE] ✓ Gemini client ready")
else:
    logger.warning("[AI‑ENGINE] ✗ Google API key missing")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPTS — ANTI-HALLUCINATION + STRICT JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_GROUNDING = (
    "CRITICAL RULES:\n"
    "1. You MUST base everything strictly on the provided document text.\n"
    "2. Do NOT use any external knowledge.\n"
    "3. Do NOT hallucinate or invent facts.\n"
    "4. Output ONLY valid JSON — no markdown fences, no commentary.\n\n"
)

EXAM_SYSTEM_PROMPT = (
    _GROUNDING +
    "You are an expert educational assessment designer.\n"
    "Generate an exam with EXACTLY 30 Multiple Choice questions AND 20 True/False questions.\n\n"
    "Output MUST be valid JSON matching this EXACT schema:\n"
    "{\n"
    '  "mcq": [\n'
    "    {\n"
    '      "id": 1,\n'
    '      "question": "...",\n'
    '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
    '      "answer": "A"\n'
    "    }\n"
    "  ],\n"
    '  "true_false": [\n'
    "    {\n"
    '      "id": 1,\n'
    '      "question": "...",\n'
    '      "answer": true\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Exactly 30 MCQ questions (id: 1-30), each with 4 options A/B/C/D, exactly 1 correct.\n"
    "- Exactly 20 True/False questions (id: 1-20), answer is boolean.\n"
    "- Use Bloom's Taxonomy: mix of Remember, Understand, Apply, Analyze.\n"
    "- Questions must cover ALL major topics in the text comprehensively.\n"
    "- All question text and options must be in the SAME language as the source text.\n"
)

MINDMAP_SYSTEM_PROMPT = (
    _GROUNDING +
    "You are a knowledge-structuring expert.\n"
    "Create a deep, premium hierarchical mind map from the provided text.\n\n"
    "Output MUST be valid JSON matching this EXACT schema:\n"
    "{\n"
    '  "id": "root",\n'
    '  "label": "Main Topic Title",\n'
    '  "children": [\n'
    "    {\n"
    '      "id": "concept-1",\n'
    '      "label": "Concept Name",\n'
    '      "children": [\n'
    "        {\n"
    '          "id": "sub-1-1",\n'
    '          "label": "Sub-concept",\n'
    '          "children": [\n'
    '            {"id": "detail-1-1-1", "label": "Detail", "children": []}\n'
    "          ]\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Minimum 3 levels deep: Root → Concepts → Sub-concepts → Details.\n"
    "- 4-8 top-level concept nodes.\n"
    "- Each concept should have 2-5 sub-concepts.\n"
    "- Max 8 words per label.\n"
    "- Use kebab-case IDs (e.g. 'concept-1', 'sub-1-2').\n"
    "- Labels must be in the SAME language as the source text.\n"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON RECOVERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor:
    1. Strip markdown code fences (```json ... ```)
    2. Extract first { ... } block
    3. Parse with json.loads
    Raises ValueError on failure with diagnostic info.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty AI response received")

    cleaned = raw_text.strip()

    # Strategy 1: Remove ```json ... ``` wrapper
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: Find the first { ... } block (greedy from first { to last })
    if not cleaned.startswith("{"):
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed. Raw (first 500 chars): {raw_text[:500]}")
        raise ValueError(f"AI returned invalid JSON: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT CHUNKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_text(text: str, chunk_size: int | None = None) -> list[str]:
    """Split text into chunks respecting sentence boundaries."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    current = ""
    sentences = re.split(r'(?<=[.!?؟。])\s+', text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROVIDER CALLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq (Llama 3) with JSON mode and temperature=0."""
    if not groq_client:
        raise ValueError("Groq API Key missing")

    logger.info(f"[AI‑ENGINE] Calling Groq ({settings.GROQ_MODEL})...")
    completion = await groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=8000,
    )
    result = completion.choices[0].message.content
    logger.info("[AI‑ENGINE] ✓ Groq call succeeded")
    return result


async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini with JSON mode and temperature=0."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Google API Key missing")

    logger.info(f"[AI‑ENGINE] Calling Gemini ({settings.GEMINI_MODEL})...")
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0,
        },
    )
    full_prompt = f"{system_prompt}\n\nUser Task:\n{user_prompt}"
    response = await asyncio.to_thread(model.generate_content, full_prompt)
    logger.info("[AI‑ENGINE] ✓ Gemini call succeeded")
    return response.text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYBRID CALL WITH FAILOVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _hybrid_call(
    system_prompt: str,
    user_prompt: str,
    primary: str = "groq",
) -> str:
    """
    Execute AI call with automatic failover.
    In 'hybrid' mode: tries primary first, then the other.
    """
    provider = settings.AI_PROVIDER

    if provider == "groq":
        callers = [("Groq", _call_groq)]
    elif provider == "gemini":
        callers = [("Gemini", _call_gemini)]
    else:  # hybrid
        if primary == "groq":
            callers = [("Groq", _call_groq), ("Gemini", _call_gemini)]
        else:
            callers = [("Gemini", _call_gemini), ("Groq", _call_groq)]

    last_error = None
    for name, caller in callers:
        try:
            return await caller(system_prompt, user_prompt)
        except Exception as e:
            last_error = e
            logger.warning(f"[AI‑ENGINE] {name} failed: {str(e)[:200]}. Trying next...")

    raise RuntimeError(f"All AI providers failed. Last error: {last_error}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATION WITH RETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_RETRIES = 2


async def generate_exam(text: str) -> ExamData:
    """
    Generate a 50-question exam (30 MCQ + 20 T/F) with retry logic.
    Uses Groq as primary, Gemini as fallback.
    """
    logger.info("[EXAM] Starting generation...")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:5])  # Use up to 5 chunks

    user_prompt = (
        f"Generate an exam from the following educational text.\n"
        f"You MUST produce exactly 30 MCQ and 20 True/False questions.\n\n"
        f"SOURCE TEXT:\n{source_text}"
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = await _hybrid_call(EXAM_SYSTEM_PROMPT, user_prompt, primary="groq")
            parsed = clean_and_parse_json(raw)
            exam = ExamData(**parsed)
            logger.info(
                f"[EXAM] ✓ Generated {len(exam.mcq)} MCQ + {len(exam.true_false)} T/F "
                f"(attempt {attempt})"
            )
            return exam
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(f"[EXAM] Attempt {attempt}/{MAX_RETRIES} failed: {e}")

    raise ValueError(f"Exam generation failed after {MAX_RETRIES} attempts: {last_error}")


async def generate_mind_map(text: str) -> MindMapNode:
    """
    Generate a deep hierarchical mind map with retry logic.
    Uses Gemini as primary, Groq as fallback.
    """
    logger.info("[MINDMAP] Starting generation...")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:3])

    user_prompt = (
        f"Create a comprehensive, deep mind map for the following text.\n\n"
        f"SOURCE TEXT:\n{source_text}"
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = await _hybrid_call(MINDMAP_SYSTEM_PROMPT, user_prompt, primary="gemini")
            parsed = clean_and_parse_json(raw)
            mind_map = MindMapNode(**parsed)
            logger.info(f"[MINDMAP] ✓ Generated (attempt {attempt})")
            return mind_map
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(f"[MINDMAP] Attempt {attempt}/{MAX_RETRIES} failed: {e}")

    raise ValueError(f"Mind map generation failed after {MAX_RETRIES} attempts: {last_error}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLE ENTRY POINT — CONCURRENT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def process_document(text: str) -> tuple[ExamData, MindMapNode]:
    """
    Run exam + mind map generation concurrently.
    Returns (ExamData, MindMapNode) tuple.
    """
    logger.info("[PROCESS] Running exam + mind map concurrently...")
    exam, mind_map = await asyncio.gather(
        generate_exam(text),
        generate_mind_map(text),
    )
    logger.info("[PROCESS] ✓ Both tasks completed")
    return exam, mind_map
