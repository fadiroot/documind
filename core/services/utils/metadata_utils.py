"""Shared utilities for metadata processing across services."""
from typing import Dict, Any, Optional


def build_resource_path(metadata: Dict[str, Any]) -> str:
    """
    Build a hierarchical resource path from document metadata.
    
    This function creates a human-readable citation path like:
    "Document Name > Part > Chapter > Article 5"
    
    Args:
        metadata: Document metadata dictionary from Azure AI Search
    
    Returns:
        Formatted resource path string for citations
    
    Example:
        >>> metadata = {
        ...     "source_document": "نظام العمل.pdf",
        ...     "legal_part_name": "الباب الخامس",
        ...     "article_reference": "المادة 9"
        ... }
        >>> build_resource_path(metadata)
        'نظام العمل > الباب الخامس > المادة 9'
    """
    # Use metadata_resource_path if available (pre-built by chunker)
    if metadata.get('metadata_resource_path'):
        return str(metadata['metadata_resource_path'])
    
    # Build from individual fields
    parts = []
    
    # 1. Document name (base level)
    source = (
        metadata.get('source_document') 
        or metadata.get('document_name') 
        or metadata.get('metadata_source_file')
    )
    if source:
        parts.append(str(source).replace('.pdf', ''))
    
    # 2. Legal Part (الباب)
    if metadata.get('legal_part_name'):
        parts.append(metadata['legal_part_name'])
    elif metadata.get('metadata_section_title') and 'الباب' in str(metadata.get('metadata_section_title', '')):
        parts.append(metadata['metadata_section_title'])
    
    # 3. Legal Chapter (الفصل)
    if metadata.get('legal_chapter_name'):
        parts.append(metadata['legal_chapter_name'])
    elif metadata.get('metadata_section_title') and 'الفصل' in str(metadata.get('metadata_section_title', '')):
        parts.append(metadata['metadata_section_title'])
    
    # 4. Article/Section reference
    if metadata.get('article_reference'):
        parts.append(metadata['article_reference'])
    elif metadata.get('article_number'):
        # Fallback: build article reference if not available
        parts.append(f"المادة {metadata['article_number']}")
    elif metadata.get('metadata_item_number') and metadata.get('metadata_item_type') == 'article':
        parts.append(f"المادة {metadata['metadata_item_number']}")
    elif metadata.get('metadata_item_title'):
        # Use item title as last resort
        parts.append(metadata['metadata_item_title'])
    
    return " > ".join(parts) if parts else "مستند غير معروف"
