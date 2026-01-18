"""Q&A endpoints using LangChain RAG chains."""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from core.models.question import QuestionRequest, QuestionResponse
from core.models.response import APIResponse
from core.models.user import UserMetadata
from core.services.agents.agent_service import AgentService
from app.routes.auth import get_current_user
from core.utils.logger import logger
import json

router = APIRouter()
agent_service = AgentService()


@router.post("/ask", response_model=APIResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: UserMetadata = Depends(get_current_user)
):
    """
    Ask a question and get an answer using LangChain RAG chains.
    
    Args:
        request: Question request with question text, optional context, and category
        current_user: Current authenticated user (from JWT token)
    
    Returns:
        API response with answer, sources, and confidence
    """
    try:
        # Use user metadata for context-aware filtering
        result = agent_service.answer_question(
            question=request.question,
            context=request.context_ids,
            category=request.category,
            user_metadata=current_user
        )
        
        response = QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence"),
            user_info_used=result.get("user_info_used", False)
        )
        
        return APIResponse(
            success=True,
            message="Question answered successfully",
            data=response.dict()
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/ask/stream")
async def ask_question_stream(
    request: QuestionRequest,
    current_user: UserMetadata = Depends(get_current_user)
):
    """
    Ask a question with streaming response.
    
    Args:
        request: Question request with optional category
        current_user: Current authenticated user (from JWT token)
    
    Returns:
        Streaming response with answer chunks
    """
    try:
        def generate():
            for chunk in agent_service.stream_answer(
                question=request.question,
                category=request.category,
                user_metadata=current_user
            ):
                # Format as SSE (Server-Sent Events)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Error streaming answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error streaming answer: {str(e)}"
        )


@router.get("/health")
async def qa_health():
    """Health check for Q&A service."""
    return {"status": "healthy", "service": "qa", "chains": "langchain_rag"}
