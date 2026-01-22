"""Result types for tool execution."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    results: List[str]
    tools_called: List[str]
    iterations: int
    success: bool
    error: Optional[str] = None
