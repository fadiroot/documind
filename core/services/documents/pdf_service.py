"""PDF processing service."""
import io
from typing import List
from core.utils.logger import logger
from core.utils.text_utils import chunk_text, clean_text
from core.utils.azure_utils import get_document_intelligence_client
from app.config import settings


class PDFService:
    """Service for extracting text from PDF files."""
    
    def __init__(self):
        self.doc_intelligence_client = get_document_intelligence_client()
    
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
    
    def chunk_pdf(self, pdf_bytes: bytes) -> List[str]:
        """
        Extract and chunk text from PDF.
        
        Args:
            pdf_bytes: PDF file as bytes
        
        Returns:
            List of text chunks
        """
        text = self.extract_text(pdf_bytes)
        chunks = chunk_text(text)
        return chunks
