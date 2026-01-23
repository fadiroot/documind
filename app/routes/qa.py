"""Q&A endpoints using LangChain RAG chains."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from core.models.question import QuestionRequest
from core.models.user import UserMetadata
from core.services.agents.agent_service import AgentService
from app.routes.auth import get_current_user
from core.utils.logger import logger
import json

router = APIRouter()
agent_service = AgentService()


@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    session_id: Optional[str] = None,
    current_user: UserMetadata = Depends(get_current_user)
):
    """Ask a question and get a streaming response (SSE).
    
    Event Types:
    - status: Progress updates
    - answer_start: Beginning of answer
    - answer_chunk: Text chunks (stream to UI)
    - answer_end: End of answer
    - complete: Final metadata (sources, user_info_used, retrieval_score)
    - error: Error messages
    """
    try:
        effective_session_id = session_id or current_user.user_id
        
        def generate():
            try:
                for event in agent_service.stream(
                    question=request.question,
                    user=current_user,
                    session_id=effective_session_id
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
