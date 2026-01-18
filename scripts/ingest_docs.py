"""Script to ingest documents into the vector store."""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.services.documents.pdf_service import PDFService
from core.services.retrieval.embedding_service import EmbeddingService
from core.services.retrieval.vectorstore_service import VectorStoreService
from core.utils.logger import logger


def ingest_document(file_path: str):
    """
    Ingest a single document.
    
    Args:
        file_path: Path to PDF file
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    pdf_service = PDFService()
    embedding_service = EmbeddingService()
    vectorstore_service = VectorStoreService()
    
    try:
        # Read PDF
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Extract and chunk
        chunks = pdf_service.chunk_pdf(pdf_bytes)
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        
        # Create embeddings
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Prepare documents
        import uuid
        from datetime import datetime
        
        document_id = str(uuid.uuid4())
        documents = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:
                doc = {
                    "id": f"{document_id}_{idx}",
                    "document_id": document_id,
                    "content": chunk,
                    "embedding": embedding,
                    "filename": os.path.basename(file_path),
                    "chunk_index": idx,
                    "metadata": {
                        "filename": os.path.basename(file_path),
                        "uploaded_at": datetime.utcnow().isoformat()
                    }
                }
                documents.append(doc)
        
        # Upload
        success = vectorstore_service.upload_documents(documents)
        
        if success:
            logger.info(f"Successfully ingested {len(documents)} chunks from {file_path}")
            return True
        else:
            logger.error(f"Failed to upload documents from {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error ingesting document {file_path}: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_docs.py <pdf_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = ingest_document(file_path)
    sys.exit(0 if success else 1)
