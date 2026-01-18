"""Text processing utilities."""
import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from app.config import settings
from core.utils.logger import logger


def chunk_text(text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, split_by_subject: bool = True) -> List[str]:
    """
    Split text into chunks. Supports subject-based splitting for structured documents.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        split_by_subject: If True, attempt to split by subject/section markers first
    
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP
    
    # Try subject-based splitting if enabled
    if split_by_subject:
        subject_chunks = _chunk_by_subject(text, chunk_size, chunk_overlap)
        if subject_chunks:
            logger.debug(f"Split text into {len(subject_chunks)} chunks by subject markers")
            return subject_chunks
    
    # Fall back to standard recursive splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def _chunk_by_subject(text: str, chunk_size: int, chunk_overlap: int) -> Optional[List[str]]:
    """
    Split text by subject/section markers (Arabic document structure).
    
    Recognizes Arabic section markers like:
    - الفصل (Chapter)
    - المادة (Article)
    - الباب (Section/Part)
    - الفقرة (Paragraph)
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunks split by subjects, or None if no markers found
    """
    # Arabic section markers patterns
    # Patterns for common Arabic document structure markers
    section_patterns = [
        r'الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+)',  # الفصل الأول, الفصل 1
        r'الفصل\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)',  # الفصل الأولى
        r'المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+)',  # المادة الأولى
        r'الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+)',  # الباب الأول
        r'##?\s+[^\n]*(?:الفصل|المادة|الباب)',  # Markdown-style headings
        r'^[^\n]*(?:الفصل|المادة|الباب)\s+[^\n]*$',  # Line starting with section marker
    ]
    
    # Combine patterns
    combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
    
    # Find all section markers - use multiple patterns for better coverage
    # Pattern 1: Lines starting with الفصل/المادة/الباب followed by ordinal or number
    pattern1 = r'^(?:#+\s*)?(?:الفصل|المادة|الباب)\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+|الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)'
    matches = list(re.finditer(pattern1, text, re.MULTILINE | re.IGNORECASE))
    
    # Pattern 2: More flexible - section markers anywhere with context
    if not matches or len(matches) < 2:
        pattern2 = r'(?:^|\n)(?:#+\s*)?(?:الفصل|المادة|الباب)\s+[^\n]+(?=\n|$)'
        matches2 = list(re.finditer(pattern2, text, re.MULTILINE | re.IGNORECASE))
        if matches2 and len(matches2) > len(matches):
            matches = matches2
    
    if not matches:
        # No section markers found, return None to use default splitting
        return None
    
    # Extract sections based on markers
    sections = []
    last_pos = 0
    
    for i, match in enumerate(matches):
        section_start = match.start()
        
        # Get section end (start of next marker or end of text)
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        # Extract section including the marker
        section = text[section_start:section_end].strip()
        if section:
            sections.append(section)
        
        last_pos = section_end
    
    # If there's text before the first match, add it as the first section
    if matches and matches[0].start() > 0:
        first_section = text[0:matches[0].start()].strip()
        if first_section:
            sections.insert(0, first_section)
    
    # Keep each subject/section as a single chunk regardless of size
    # Subjects (الفصل, المادة, etc.) are semantic units and should be preserved intact
    # Only split extremely large sections (e.g., > 10x chunk_size) to avoid embedding limits
    final_chunks = []
    max_subject_size = chunk_size * 10  # Allow subjects up to 10x normal chunk size
    
    for section in sections:
        if len(section) <= max_subject_size:
            # Keep subject as single chunk (even if > chunk_size)
            final_chunks.append(section)
        else:
            # Only split extremely large subjects (> 10x chunk_size)
            # This is rare but prevents embedding token limit issues
            logger.warning(f"Subject exceeds {max_subject_size} chars ({len(section)}), splitting it")
            sub_sections = _split_large_section(section, chunk_size, chunk_overlap)
            final_chunks.extend(sub_sections)
    
    return final_chunks if final_chunks else None


def _split_large_section(section: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split a large section into smaller chunks while preserving structure.
    
    Tries to split by paragraphs, then sentences, then falls back to character-based.
    
    Args:
        section: Section text to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked sections
    """
    # Try splitting by double newlines (paragraphs)
    paragraphs = section.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk size
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself is too large, split it further
            if len(para) > chunk_size:
                # Split by sentences
                sentences = re.split(r'[.!?]\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk = para
        else:
            # Add paragraph to current chunk
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # If still too large, use recursive splitter as fallback
    if not chunks or any(len(chunk) > chunk_size * 1.5 for chunk in chunks):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(section)
    
    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()
