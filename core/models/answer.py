"""Answer response models."""
from pydantic import BaseModel
from typing import List, Optional


class AnswerItem(BaseModel):
    """Single answer item with content and resource reference."""
    content: str
    resource: str  # Document path, article reference, or subject


class AnswerResponse(BaseModel):
    """Structured answer response from LLM."""
    answers: List[AnswerItem]
