"""Prompt builder for RAG system - formats context and questions."""
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from core.services.utils.metadata_utils import build_resource_path
from core.utils.logger import logger


class PromptBuilder:
    """Simple prompt builder - formats context and question."""
    
    def __init__(self):
        """Initialize prompt builder with prompts directory."""
        self._prompts_dir = Path(__file__).parent / "templates"
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt text from .promptly file, extracting content after YAML frontmatter."""
        prompt_path = self._prompts_dir / filename
        try:
            content = prompt_path.read_text(encoding="utf-8")
            # Extract content after YAML frontmatter (after ---\n---\n)
            if "---\n" in content:
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    return parts[2].strip()
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt {filename}: {str(e)}")
            return ""
    
    def build_system_prompt(self) -> str:
        """Build system prompt."""
        return self._load_prompt("system_prompt.promptly")
    
    def build_context_prompt(self, docs: List[Document]) -> str:
        """Build context from retrieved documents."""
        if not docs:
            return "No relevant documents were found."
        
        template = self._load_prompt("rag_context_template.promptly")
        separator = "─" * 50
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            metadata = doc.metadata or {}
            resource_path = self._get_resource_path(metadata)
            context_part = template.format(
                index=i,
                resource_path=resource_path,
                separator=separator,
                content=content
            )
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def build_user_prompt(self, question: str, history_summary: Optional[str] = None) -> str:
        """Build user prompt."""
        template = self._load_prompt("user_prompt_template.promptly")
        
        history = history_summary if history_summary and history_summary != "No previous conversation." else ""
        
        return template.format(
            history=history,
            question=question
        ).strip()
    
    def build_full_prompt(
        self,
        question: str,
        docs: List[Document],
        history_summary: Optional[str] = None
    ) -> Tuple[str, str]:
        """Build complete prompt: (system_prompt, full_prompt)."""
        system_prompt = self.build_system_prompt()
        context = self.build_context_prompt(docs)
        user_prompt = self.build_user_prompt(question, history_summary)
        
        full_prompt = f"{context}\n\n{user_prompt}"
        return system_prompt, full_prompt
    
    def _get_resource_path(self, metadata: dict) -> str:
        """Get resource path from metadata using shared utility."""
        return build_resource_path(metadata)
