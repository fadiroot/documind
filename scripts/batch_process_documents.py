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
from core.services.retrieval.vectorstore_service import VectorStoreService
from core.utils.logger import logger


def process_pdf_file(
    file_path: Path,
    pdf_service: PDFService,
    embedding_service: EmbeddingService,
    vectorstore_service: VectorStoreService,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Process a single PDF file through the full pipeline.
    
    Args:
        file_path: Path to PDF file
        pdf_service: PDF service instance
        embedding_service: Embedding service instance
        vectorstore_service: Vector store service instance
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
        
        # Extract and chunk text
        logger.info(f"  Extracting text and chunking...")
        chunks = pdf_service.chunk_pdf(pdf_bytes)
        
        if not chunks:
            result["error"] = "No text extracted from PDF"
            logger.warning(f"  ⚠️  {result['error']}")
            return result
        
        result["chunks_processed"] = len(chunks)
        logger.info(f"  ✓ Extracted {len(chunks)} chunks")
        
        # Create embeddings
        logger.info(f"  Creating embeddings...")
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Count successful embeddings
        valid_embeddings = sum(1 for emb in embeddings if emb is not None)
        if valid_embeddings == 0:
            result["error"] = "Failed to create embeddings"
            logger.error(f"  ✗ {result['error']}")
            return result
        
        logger.info(f"  ✓ Created {valid_embeddings}/{len(chunks)} embeddings")
        
        # Prepare documents for vector store
        document_id = str(uuid.uuid4())
        filename = file_path.name
        documents = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # Only include if embedding was created
                # Approximate token count (rough estimate: ~4 chars per token)
                token_count = len(chunk.split())
                
                doc = {
                    "id": f"{document_id}_{idx}",
                    "content": chunk,
                    "contentVector": embedding,
                    "document_name": filename,
                    "page_number": 1,
                    "chunk_index": idx,
                    "token_count": token_count
                }
                documents.append(doc)
        
        # Upload to vector store
        logger.info(f"  Uploading {len(documents)} documents to index...")
        success = vectorstore_service.upload_documents(documents)
        
        if success:
            result["success"] = True
            result["chunks_uploaded"] = len(documents)
            logger.info(f"  ✓ Successfully uploaded {len(documents)} chunks")
        else:
            result["error"] = "Failed to upload documents to vector store"
            logger.error(f"  ✗ {result['error']}")
        
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
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info("=" * 60)
    
    # Initialize services
    pdf_service = PDFService()
    embedding_service = EmbeddingService()
    vectorstore_service = VectorStoreService()
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    total_chunks = 0
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        logger.info("-" * 60)
        
        result = process_pdf_file(
            pdf_file,
            pdf_service,
            embedding_service,
            vectorstore_service,
            skip_existing
        )
        
        results.append(result)
        
        if result["success"]:
            successful += 1
            total_chunks += result["chunks_uploaded"]
        else:
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files found:     {len(pdf_files)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed:                {failed}")
    logger.info(f"Total chunks uploaded: {total_chunks}")
    logger.info("=" * 60)
    
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
