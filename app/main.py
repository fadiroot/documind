"""Main FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import qa, auth
from core.utils.logger import logger

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)


@app.on_event("startup")
async def startup_event():
    """Log configuration on startup."""
    logger.info("DocuMind RAG Application Starting...")
    logger.info(f"Azure OpenAI: {'Configured' if settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_API_KEY else 'Not Configured'}")
    logger.info(f"Azure AI Search: {'Configured' if (settings.AZURE_AI_SEARCH_ENDPOINT or settings.AZURE_SEARCH_ENDPOINT) else 'Not Configured'}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(qa.router, prefix="/api/qa", tags=["Q&A"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Application API",
        "version": settings.API_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
