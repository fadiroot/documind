"""LangChain agent factory - creates agent with tools from agent_chain dependencies."""
from pathlib import Path
from typing import Optional

from langchain.agents import create_agent

from core.services.agents.azure_client import get_azure_llm
from core.services.retrieval.retrieval_service import RetrievalService, create_retrieve_tool


def _load_system_prompt(filename: str = "system_prompt.promptly") -> str:
    """Load prompt from templates (extracted from agent_chain)."""
    prompts_dir = Path(__file__).parent.parent / "prompts" / "templates"
    try:
        content = (prompts_dir / filename).read_text(encoding="utf-8")
        if "---\n" in content:
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content.strip()
    except Exception:
        return "You are a helpful AI assistant for Saudi Arabian legal documents."


def create_documind_agent(
    min_retrieval_score: float = 0.3,
):
    """
    Create a LangChain agent with retrieve_tool, using model and prompts from agent_chain.
    """
    llm = get_azure_llm()
    if not llm:
        raise RuntimeError("Azure OpenAI not configured")

    retrieval_service = RetrievalService(
        min_score_threshold=min_retrieval_score,
        enable_reranking=False,
    )
    tools = [create_retrieve_tool(retrieval_service)]
    system_prompt = _load_system_prompt()

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
