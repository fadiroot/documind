"""PDF processing service."""
import io
from typing import List, Optional
from core.utils.logger import logger
from core.utils.text_utils import clean_text
from core.utils.azure_utils import get_document_intelligence_client
from core.services.documents.chunker import DocumentChunker, DocumentChunk
from app.config import settings


class PDFService:
    """Service for extracting text from PDF files."""
    
    def __init__(self):
        """Initialize PDF service with document chunker."""
        self.doc_intelligence_client = get_document_intelligence_client()
        
        chunk_size = getattr(settings, 'CHUNK_SIZE', 2000)
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 200)
        
        self.chunker = DocumentChunker(
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF using Azure Document Intelligence.
        
        Args:
            pdf_bytes: PDF file as bytes
        
        Returns:
            Extracted text
        
        Raises:
            ValueError: If Azure Document Intelligence client is not available
        """
        if not self.doc_intelligence_client:
            raise ValueError("Azure Document Intelligence client is not available. Please configure Azure credentials.")
        
        try:
            # For Azure Document Intelligence SDK 1.0+, use body parameter with IO stream
            with io.BytesIO(pdf_bytes) as file_stream:
                poller = self.doc_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-read",
                    body=file_stream,
                    content_type="application/pdf"
                )
                result = poller.result()
                
                text_parts = []
                if result.content:
                    text_parts.append(result.content)
                
                return clean_text("\n".join(text_parts))
        except Exception as e:
            logger.error(f"Error extracting text with Azure: {str(e)}")
            raise
    
    def chunk_pdf(self, pdf_bytes: bytes, filename: Optional[str] = None) -> List[str]:
        """
        Extract and chunk text from PDF (simple text chunks).
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Optional filename for metadata
        
        Returns:
            List of text chunks
        """
        text = self.extract_text(pdf_bytes)
        from core.utils.text_utils import chunk_text
        return chunk_text(text)
    
    def chunk_pdf_with_metadata(
        self, 
        pdf_bytes: bytes, 
        filename: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Extract and chunk text from PDF with metadata.
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Optional filename for metadata
        
        Returns:
            List of DocumentChunk objects with metadata
        """
        text = self.extract_text(pdf_bytes)
        source_file = filename or "unknown"
        return self.chunker.chunk_document(text, source_file)