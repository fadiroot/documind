"""Response data models."""
from pydantic import BaseModel
from typing import Optional, Any


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
