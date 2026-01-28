import io
from typing import List, Optional
from core.utils.logger import logger
from core.utils.text_utils import clean_text
from core.utils.azure_utils import get_document_intelligence_client
from core.services.documents.chunker import DocumentChunker, DocumentChunk
from app.config import settings


class PDFService:
    def __init__(self):
        self.doc_intelligence_client = get_document_intelligence_client()
        
        chunk_size = getattr(settings, 'CHUNK_SIZE', 2000)
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 200)
        
        self.chunker = DocumentChunker(
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_text(self, pdf_bytes: bytes) -> str:
        if not self.doc_intelligence_client:
            raise ValueError("Azure Document Intelligence client is not available. Please configure Azure credentials.")
        
        try:
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
        text = self.extract_text(pdf_bytes)
        from core.utils.text_utils import chunk_text
        return chunk_text(text)
    
    def chunk_pdf_with_metadata(
        self, 
        pdf_bytes: bytes, 
        filename: Optional[str] = None
    ) -> List[DocumentChunk]:
        text = self.extract_text(pdf_bytes)
        source_file = filename or "unknown"
        return self.chunker.chunk_document(text, source_file)
