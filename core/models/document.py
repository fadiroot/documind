"""Document data models."""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentBase(BaseModel):
    """Base document model."""
    content: str
    metadata: Optional[dict] = None


class DocumentCreate(DocumentBase):
    """Document creation model."""
    filename: str
    file_type: Optional[str] = None


class Document(DocumentBase):
    """Document model with ID."""
    id: str
    filename: str
    file_type: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentChunk(BaseModel):
    """Document chunk model."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True
