#!/usr/bin/env python3
"""Batch script to process all PDF documents in a folder."""
import sys
import os
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.services.documents.pdf_service import PDFService
from core.services.retrieval.embedding_service import EmbeddingService
from core.services.indexing.storage_service import StorageService
from core.utils.logger import logger


def process_pdf_file(
    file_path: Path,
    pdf_service: PDFService,
    embedding_service: EmbeddingService,
    storage_service: StorageService,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Process a single PDF file through the full pipeline.
    
    Args:
        file_path: Path to PDF file
        pdf_service: PDF service instance
        embedding_service: Embedding service instance
        storage_service: Storage service instance for uploading documents
        skip_existing: Whether to skip files that already exist in index
    
    Returns:
        Dictionary with processing results
    """
    result = {
        "file": str(file_path),
        "success": False,
        "chunks_processed": 0,
        "chunks_uploaded": 0,
        "error": None
    }
    
    try:
        # Read PDF file
        logger.info(f"Processing: {file_path.name}")
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Extract and chunk text with flexible metadata
        filename = file_path.name
        document_chunks = pdf_service.chunk_pdf_with_metadata(pdf_bytes, filename=filename)
        
        if not document_chunks:
            result["error"] = "No text extracted from PDF"
            logger.warning(f"  ⚠️  {result['error']}")
            return result
        
        result["chunks_processed"] = len(document_chunks)
        logger.info(f"  ✓ {len(document_chunks)} chunks extracted")
        
        # Extract content for embeddings
        chunk_contents = [chunk.content for chunk in document_chunks]
        
        # Create embeddings
        embeddings = embedding_service.create_embeddings(chunk_contents)
        
        # Count successful embeddings
        valid_embeddings = sum(1 for emb in embeddings if emb is not None)
        if valid_embeddings == 0:
            result["error"] = "Failed to create embeddings"
            logger.error(f"  ✗ {result['error']}")
            return result
        
        logger.info(f"  ✓ {valid_embeddings} embeddings created")
        
        # Prepare documents for vector store with flexible metadata
        document_id = str(uuid.uuid4())
        documents = []
        chunk_counter = 0
        
        for idx, (doc_chunk, embedding) in enumerate(zip(document_chunks, embeddings)):
            if embedding:  # Only include if embedding was created
                chunk_counter += 1
                # Build section title from available metadata
                section_title_parts = []
                if doc_chunk.item_title:
                    section_title_parts.append(doc_chunk.item_title)
                section_title = " - ".join(section_title_parts) if section_title_parts else doc_chunk.section_title
                
                # Approximate token count
                token_count = len(doc_chunk.content.split())
                
                # Get resource path from chunk metadata (now built in chunker with full hierarchy)
                resource_path = doc_chunk.metadata.get('resource_path')
                if not resource_path:
                    # Fallback: build basic path if not available
                    resource_path = filename.replace('.pdf', '')
                    if doc_chunk.article_reference:
                        resource_path = f"{resource_path} > {doc_chunk.article_reference}"
                
                # Build document dictionary with all fields
                # Filter out None values to avoid Azure Search issues
                doc = {
                    "id": f"{document_id}_{idx}",
                    "content": doc_chunk.content,
                    "contentVector": embedding,
                    # Required index fields
                    "source_document": filename,
                    # Metadata fields
                    "metadata_section_title": section_title,
                    "metadata_resource_path": resource_path,
                    # Legacy fields for backward compatibility
                    "document_name": filename,
                    "page_number": doc_chunk.page_number or 1,
                    "chunk_index": doc_chunk.chunk_index or idx,
                    "token_count": token_count,
                }
                
                # Add optional fields only if they have values (to avoid None issues)
                # Legal hierarchy (now properly tracked by hierarchy context in chunker)
                if doc_chunk.legal_part_name:
                    doc["legal_part_name"] = doc_chunk.legal_part_name
                if doc_chunk.legal_chapter_name:
                    doc["legal_chapter_name"] = doc_chunk.legal_chapter_name
                
                # Article reference (now in Arabic) and number
                if doc_chunk.article_reference:
                    doc["article_reference"] = doc_chunk.article_reference
                if doc_chunk.item_number and doc_chunk.item_type == "article":
                    doc["article_number"] = doc_chunk.item_number
                
                # Classification fields
                if doc_chunk.category:
                    doc["category"] = doc_chunk.category
                if doc_chunk.target_audience:
                    doc["target_audience"] = doc_chunk.target_audience
                
                # Keywords extracted using KeyBERT
                if doc_chunk.keywords:
                    doc["keywords"] = doc_chunk.keywords
                
                # Document metadata
                if doc_chunk.document_title:
                    doc["document_title"] = doc_chunk.document_title
                
                # Hierarchical structure metadata
                if doc_chunk.item_number:
                    doc["metadata_item_number"] = doc_chunk.item_number
                if doc_chunk.item_type:
                    doc["metadata_item_type"] = doc_chunk.item_type
                if doc_chunk.item_title:
                    doc["metadata_item_title"] = doc_chunk.item_title
                
                # Additional metadata from metadata dict
                for k, v in doc_chunk.metadata.items():
                    if v:  # Only include non-empty values
                        # Map common metadata keys to index fields
                        if k == "paragraph_number":
                            doc["paragraph_number"] = v
                        elif k == "clause_number":
                            doc["clause_number"] = v
                        elif k == "procedure_name":
                            doc["procedure_name"] = v
                        elif k == "procedure_step":
                            doc["procedure_step"] = v
                        elif k == "policy_name":
                            doc["policy_name"] = v
                        elif k == "annex_name":
                            doc["annex_name"] = v
                        elif k == "rank":
                            doc["rank"] = v
                        elif k == "grade":
                            doc["grade"] = v
                        elif k == "category_class":
                            doc["category_class"] = v
                        elif k == "group":
                            doc["group"] = v
                        elif k == "cadre_classification":
                            doc["cadre_classification"] = v
                
                # Final cleanup: Remove all None/null values to keep index clean
                doc_clean = {k: v for k, v in doc.items() if v is not None and v != "" and v != []}
                documents.append(doc_clean)
        
        # Upload to search index
        success = storage_service.upload_documents(documents)
        
        if success:
            result["success"] = True
            result["chunks_uploaded"] = len(documents)
            logger.info(f"  ✓ {len(documents)} chunks uploaded")
        else:
            result["error"] = "Failed to upload"
            logger.error(f"  ✗ Upload failed")
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ✗ Error processing {file_path.name}: {str(e)}")
        return result


def batch_process_folder(
    folder_path: str,
    skip_existing: bool = False,
    recursive: bool = False,
    file_pattern: str = "*.pdf"
) -> Dict[str, Any]:
    """
    Process all PDF files in a folder.
    
    Args:
        folder_path: Path to folder containing PDFs
        skip_existing: Whether to skip files that already exist in index
        recursive: Whether to search subdirectories recursively
        file_pattern: File pattern to match (default: *.pdf)
    
    Returns:
        Dictionary with batch processing results
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return {"success": False, "error": "Folder not found"}
    
    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        return {"success": False, "error": "Path is not a directory"}
    
    # Find all PDF files
    if recursive:
        pdf_files = list(folder.rglob(file_pattern))
    else:
        pdf_files = list(folder.glob(file_pattern))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return {
            "success": True,
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "results": []
        }
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Initialize services
    pdf_service = PDFService()
    embedding_service = EmbeddingService()
    storage_service = StorageService()
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    total_chunks = 0
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"[{idx}/{len(pdf_files)}] {pdf_file.name}")
        
        result = process_pdf_file(
            pdf_file,
            pdf_service,
            embedding_service,
            storage_service,
            skip_existing
        )
        
        results.append(result)
        
        if result["success"]:
            successful += 1
            total_chunks += result["chunks_uploaded"]
        else:
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Processed: {successful}/{len(pdf_files)} files | {total_chunks} chunks")
    if failed > 0:
        logger.warning(f"✗ Failed: {failed} files")
    logger.info(f"{'='*60}")
    
    return {
        "success": True,
        "total_files": len(pdf_files),
        "processed": successful,
        "failed": failed,
        "total_chunks": total_chunks,
        "results": results
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process PDF documents from a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in a folder
  python batch_process_documents.py /path/to/documents
  
  # Process recursively including subdirectories
  python batch_process_documents.py /path/to/documents --recursive
  
  # Skip files that already exist in index
  python batch_process_documents.py /path/to/documents --skip-existing
        """
    )
    
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing PDF documents"
    )
    
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search subdirectories recursively"
    )
    
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Skip files that already exist in index"
    )
    
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*.pdf",
        help="File pattern to match (default: *.pdf)"
    )
    
    args = parser.parse_args()
    
    # Process folder
    result = batch_process_folder(
        args.folder,
        skip_existing=args.skip_existing,
        recursive=args.recursive,
        file_pattern=args.pattern
    )
    
    # Exit with appropriate code
    if result.get("success") and result.get("failed", 0) == 0:
        sys.exit(0)
    elif result.get("success"):
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # Complete failure


if __name__ == "__main__":
    main()
