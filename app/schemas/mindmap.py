from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request ──────────────────────────────────────────────────────────────────

class MindMapRequest(BaseModel):
    """Request body for mind map generation."""
    text: str = Field(..., min_length=20, description="Educational text to summarize as a mind map")


# ── Response ─────────────────────────────────────────────────────────────────

class MindMapNode(BaseModel):
    """A single node in the mind map tree (recursive)."""
    id: str
    label: str
    children: Optional[List[MindMapNode]] = []


class MindMapResponse(BaseModel):
    """Full mind map response returned to the client."""
    root_node: MindMapNode
