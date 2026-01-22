"""Result types for retrieval operations."""
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """Structured result from retrieval operation."""
    documents: List[Document]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    total_found: int
    filtered_count: int  # Number filtered by thresholds
    
    def has_results(self) -> bool:
        """Check if retrieval found any documents."""
        return len(self.documents) > 0
    
    def get_average_score(self) -> float:
        """Get average relevance score."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)
    
    def get_max_score(self) -> float:
        """Get maximum relevance score."""
        return max(self.scores) if self.scores else 0.0
