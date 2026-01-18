"""User data models."""
from pydantic import BaseModel
from typing import Optional


class User(BaseModel):
    """User model with metadata."""
    user_id: str
    username: str
    password: str  # Hashed in production
    full_name: str
    cadre: str
    current_rank: Optional[str] = None
    years_in_rank: Optional[int] = None
    job_title: Optional[str] = None
    administration: str
    target_source_file: Optional[str] = None
    expected_filter: Optional[str] = None


class UserMetadata(BaseModel):
    """User metadata for context (without sensitive info)."""
    user_id: str
    full_name: str
    cadre: str
    current_rank: Optional[str] = None
    years_in_rank: Optional[int] = None
    job_title: Optional[str] = None
    administration: str
    target_source_file: Optional[str] = None
    expected_filter: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    user: UserMetadata
