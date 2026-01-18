"""Document upload and processing endpoints."""
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from core.models.response import APIResponse, ErrorResponse
from core.models.document import Document, DocumentCreate
from core.services.documents.pdf_service import PDFService
from core.services.retrieval.embedding_service import EmbeddingService
from core.services.retrieval.vectorstore_service import VectorStoreService
from core.utils.logger import logger

router = APIRouter()
pdf_service = PDFService()
embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService()


@router.post("/upload", response_model=APIResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
    
    Returns:
        API response with document ID and processing status
    """
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file content
        pdf_bytes = await file.read()
        
        # Extract and chunk text
        chunks = pdf_service.chunk_pdf(pdf_bytes)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF"
            )
        
        # Create document ID
        document_id = str(uuid.uuid4())
        
        # Create embeddings for chunks
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Prepare documents for vector store
        documents = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # Only include if embedding was created
                doc = {
                    "id": f"{document_id}_{idx}",
                    "content": chunk,
                    "contentVector": embedding,
                    "document_name": file.filename,
                    "page_number": 1,
                    "chunk_index": idx,
                    "token_count": len(chunk.split())  # Approximate token count
                }
                documents.append(doc)
        
        # Upload to vector store
        success = vectorstore_service.upload_documents(documents)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload documents to vector store"
            )
        
        return APIResponse(
            success=True,
            message=f"Document processed successfully. {len(documents)} chunks uploaded.",
            data={
                "document_id": document_id,
                "filename": file.filename,
                "chunks_count": len(documents)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.delete("/{document_id}", response_model=APIResponse)
async def delete_document(document_id: str):
    """
    Delete a document from the vector store.
    
    Args:
        document_id: ID of the document to delete
    
    Returns:
        API response with deletion status
    """
    try:
        success = vectorstore_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        return APIResponse(
            success=True,
            message=f"Document {document_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/health")
async def docs_health():
    """Health check for document service."""
    return {"status": "healthy", "service": "docs"}
