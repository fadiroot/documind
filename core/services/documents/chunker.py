"""Simple document chunker with pattern-based metadata extraction."""
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from core.utils.logger import logger
from core.services.documents.arabic_number_parser import ArabicNumberParser
from core.services.documents.classification_scorer import ClassificationScorer
from core.services.documents.keyword_extractor import KeywordExtractor


@dataclass
class DocumentChunk:
    """
    Document chunk structure with metadata for RAG system.
    """
    content: str
    # Document identification
    document_name: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    
    
    # Numbered items
    item_number: Optional[str] = None
    item_type: Optional[str] = None  # "article", "section", "paragraph", etc.
    article_reference: Optional[str] = None  # e.g., "المادة 151" (in Arabic)
    
    # Title/heading
    section_title: Optional[str] = None
    item_title: Optional[str] = None
    
    # Hierarchy context (for legal documents)
    legal_part_name: Optional[str] = None  # الباب
    legal_chapter_name: Optional[str] = None  # الفصل
    
    # Classification fields
    category: Optional[str] = None  # Leave, Financial Rights, Discipline, etc.
    target_audience: Optional[str] = None  # Engineers, General Civil Servants, etc.
    
    # Keywords extracted using KeyBERT
    keywords: Optional[List[str]] = None
    
    # Additional metadata
    metadata: Dict[str, str] = field(default_factory=dict)
    
    # Position information
    chunk_index: Optional[int] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None


@dataclass
class HierarchyContext:
    """Tracks current position in document hierarchy."""
    document_title: Optional[str] = None
    current_part: Optional[str] = None  # الباب
    current_part_number: Optional[str] = None
    current_chapter: Optional[str] = None  # الفصل
    current_chapter_number: Optional[str] = None
    current_article_number: Optional[str] = None


class DocumentChunker:
    """
    Simple document chunker that splits by headers/titles and extracts metadata using patterns.
    
    Approach:
    1. Simple text splitting by headers (المادة, الباب, الفصل, numbered sections)
    2. Pattern-based metadata extraction (category, target_audience, article_reference, etc.)
    3. Hierarchical context tracking to properly fill legal_part_name, legal_chapter_name
    4. All metadata values in Arabic
    """
    
    # Improved header patterns - only match structural headers, not in-sentence usage
    # Patterns match headers at start of line (using ^ with MULTILINE flag)
    # Note: We avoid variable-width lookbehind which Python doesn't support
    HEADER_PATTERNS = [
        # Part pattern: matches "الباب" at line start - HIGHEST PRIORITY
        r'^\s*الباب\s+',
        # Chapter pattern: matches "الفصل" at line start
        r'^\s*الفصل\s+',
        # Article pattern: matches "المادة" followed by number/ordinal at line start
        # Uses ^ anchor with MULTILINE flag to match start of line
        r'^\s*المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?)',
        # Numbered sections: matches "1.", "2)", etc. at line start
        r'^\s*\d+[\.\)]\s+',
    ]
    
    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Initialize document chunker.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters (default: 1500 for ~500-750 tokens)
            chunk_overlap: Overlap between chunks in characters
        """
        # Ensure max_chunk_size is reasonable (embedding models typically have 8K token limit)
        # For Arabic: ~2-3 chars per token, so 1500 chars ≈ 500-750 tokens (safe margin)
        self.max_chunk_size = min(max_chunk_size, 2000)  # Cap at 2000 for safety
        self.chunk_overlap = chunk_overlap
        self._number_parser = ArabicNumberParser()
        self._classifier = ClassificationScorer()
        self._keyword_extractor = KeywordExtractor(top_n=5)
        self._hierarchy_context = HierarchyContext()
    
    def chunk_document(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Chunk document by splitting at headers, then extract metadata using patterns.
        
        Args:
            text: Full text of the document
            source_file: Name of the source file
        
        Returns:
            List of DocumentChunk objects with metadata
        """
        # Normalize text
        text = self._normalize_text(text)
        
        # Extract document title from beginning
        document_title = self._extract_title(text)
        
        # Initialize hierarchy context
        self._hierarchy_context = HierarchyContext(document_title=document_title)
        
        # Split text by headers
        blocks = self._split_by_headers(text)
        
        # Create chunks with basic structure, splitting large blocks if needed
        chunks = []
        chunk_idx = 0
        for header, content in blocks:
            # Update hierarchy context based on header
            self._update_hierarchy_context(header, content)
            
            # Remove header from content if it's duplicated (prevent duplication)
            content = self._remove_header_duplication(header, content)
            
            # If content is too large, split it further
            if len(content) > self.max_chunk_size:
                sub_blocks = self._split_large_block(content)
                for sub_content in sub_blocks:
                    # Preserve header context for split chunks
                    full_content = self._build_content_with_context(header, sub_content, document_title)
                    # Use full content (with header) for metadata extraction to preserve context
                    chunk = self._create_chunk(
                        full_content, source_file, document_title, 
                        header, sub_content, chunk_idx  # Use sub_content for extraction
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                # Content fits in one chunk
                full_content = self._build_content_with_context(header, content, document_title)
                chunk = self._create_chunk(
                    full_content, source_file, document_title,
                    header, content, chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1
        
        return chunks
    
    def _update_hierarchy_context(self, header: str, content: str):
        """
        Update hierarchy context based on current header.
        Tracks الباب (Part), الفصل (Chapter), and المادة (Article) for proper metadata.
        """
        # Check for الباب (Part)
        if "الباب" in header:
            # Extract full part reference (e.g., "الباب الخامس" or "الباب 5")
            part_match = re.search(
                r'(الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))',
                header
            )
            if part_match:
                part_ref = part_match.group(1)
                
                # Extract part name/title (text after the part reference)
                title_match = re.search(
                    r'الباب\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    header + "\n" + content[:200]
                )
                if title_match:
                    part_title = title_match.group(1).strip()
                    # Use full reference + title
                    self._hierarchy_context.current_part = f"{part_ref}: {part_title}"
                else:
                    # Just use the part reference
                    self._hierarchy_context.current_part = part_ref
                
                # Reset chapter when entering new part
                self._hierarchy_context.current_chapter = None
                self._hierarchy_context.current_chapter_number = None
        
        # Check for الفصل (Chapter)
        if "الفصل" in header:
            # Extract full chapter reference (e.g., "الفصل الأول" or "الفصل 1")
            chapter_match = re.search(
                r'(الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))',
                header
            )
            if chapter_match:
                chapter_ref = chapter_match.group(1)
                
                # Extract chapter name/title (text after the chapter reference)
                title_match = re.search(
                    r'الفصل\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    header + "\n" + content[:200]
                )
                if title_match:
                    chapter_title = title_match.group(1).strip()
                    # Use full reference + title
                    self._hierarchy_context.current_chapter = f"{chapter_ref}: {chapter_title}"
                else:
                    # Just use the chapter reference
                    self._hierarchy_context.current_chapter = chapter_ref
        
        # Check for المادة (Article)
        if "المادة" in header:
            article_number = self._number_parser.parse_article_number(header)
            if article_number:
                self._hierarchy_context.current_article_number = article_number
    
    def _create_chunk(
        self, 
        full_content: str, 
        source_file: str, 
        document_title: Optional[str],
        header: str,
        content: str,
        chunk_idx: int
    ) -> DocumentChunk:
        """Create a chunk with metadata extraction."""
        chunk = DocumentChunk(
            content=full_content.strip(),
            document_name=source_file,
            document_title=document_title,
            chunk_index=chunk_idx
        )
        
        # Add hierarchy context to chunk
        chunk.legal_part_name = self._hierarchy_context.current_part
        chunk.legal_chapter_name = self._hierarchy_context.current_chapter
        
        # Extract metadata using pattern-based techniques
        self._extract_metadata(chunk, header, content, document_title)
        
        return chunk
    
    def _split_large_block(self, content: str) -> List[str]:
        """
        Split a large block into smaller chunks.
        Tries paragraphs first, then sentences.
        """
        # Try splitting by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n+', content)
        
        if len(paragraphs) > 1:
            # Split by paragraphs
            sub_blocks = []
            current_block = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph exceeds limit, save current and start new
                if current_block and len(current_block) + len(para) + 2 > self.max_chunk_size:
                    sub_blocks.append(current_block)
                    # Add overlap
                    overlap = current_block[-self.chunk_overlap:] if len(current_block) > self.chunk_overlap else current_block
                    current_block = overlap + "\n\n" + para
                else:
                    current_block += "\n\n" + para if current_block else para
            
            if current_block:
                sub_blocks.append(current_block)
            
            # If still too large, split by sentences
            final_blocks = []
            for block in sub_blocks:
                if len(block) > self.max_chunk_size:
                    final_blocks.extend(self._split_by_sentences(block))
                else:
                    final_blocks.append(block)
            
            return final_blocks if final_blocks else [content]
        else:
            # No paragraphs found, split by sentences
            return self._split_by_sentences(content)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        # Split by sentence endings
        sentences = re.split(r'([.!?]\s+)', text)
        
        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])
        
        if len(sentences) % 2 == 1:
            combined_sentences.append(sentences[-1])
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in combined_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                chunks.append(current_chunk)
                # Add overlap
                overlap_sentences = current_chunk.split('.')[-3:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title from first few lines."""
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            # Look for title indicators
            if any(keyword in line for keyword in ['لائحة', 'نظام', 'دليل', 'تعليمات', 'قرار']):
                if 20 <= len(line) <= 200:
                    return line
        return None
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text by headers (المادة, الباب, الفصل, numbered sections).
        Returns list of (header, content) tuples.
        
        Improved to only match structural headers, not in-sentence usage.
        """
        blocks = []
        
        # Find all header positions with validation
        header_positions = []
        for pattern in self.HEADER_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Validate it's a structural header (not in middle of sentence)
                if self._is_structural_header(match, text):
                    header_positions.append((match.start(), match.group(0).strip()))
        
        # Remove duplicates and sort by position
        header_positions = list(dict.fromkeys(header_positions))  # Preserve order, remove duplicates
        header_positions.sort(key=lambda x: x[0])
        
        if not header_positions:
            # No headers found, split by paragraphs to avoid huge chunks
            if len(text) > self.max_chunk_size:
                # Split into paragraphs
                paragraphs = re.split(r'\n\s*\n+', text)
                blocks = []
                current_block = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if current_block and len(current_block) + len(para) + 2 > self.max_chunk_size:
                        blocks.append(("", current_block))
                        current_block = para
                    else:
                        current_block += "\n\n" + para if current_block else para
                if current_block:
                    blocks.append(("", current_block))
                return blocks if blocks else [("", text)]
            return [("", text)]
        
        # Extract blocks between headers
        for i, (pos, header) in enumerate(header_positions):
            start = pos
            end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(text)
            content = text[start:end].strip()
            # Extract header text (first line or until newline)
            header_text = header
            if '\n' in content[:100]:
                first_line_end = content.find('\n')
                potential_header = content[:first_line_end].strip()
                if any(pattern_word in potential_header for pattern_word in ['المادة', 'الباب', 'الفصل']):
                    header_text = potential_header
                    content = content[first_line_end:].strip()
            blocks.append((header_text, content))
        
        return blocks
    
    def _is_structural_header(self, match: re.Match, text: str) -> bool:
        """
        Validate that a match is a structural header, not in-sentence usage.
        
        A structural header should:
        1. Be at start of line (after newline or start of text)
        2. Not be in middle of a sentence
        3. Be followed by content (not end of text immediately)
        """
        start_pos = match.start()
        
        # Check if it's at start of line
        if start_pos > 0:
            char_before = text[start_pos - 1]
            if char_before not in ['\n', '\r']:
                return False
        
        # Check if there's content after (not just end of text)
        end_pos = match.end()
        if end_pos >= len(text):
            return False
        
        # Check that it's not in middle of a sentence (no period/question/exclamation before)
        text_before = text[max(0, start_pos - 50):start_pos]
        if re.search(r'[.!?]\s*$', text_before):
            # Might be in sentence, but allow if it's clearly a header pattern
            pass
        
        return True
    
    def _remove_header_duplication(self, header: str, content: str) -> str:
        """
        Remove header text from content if it's duplicated.
        Prevents header from appearing both in header field and content.
        """
        if not header:
            return content
        
        # Clean header for comparison
        header_clean = header.strip()
        
        # Check if header appears at start of content
        content_start = content[:len(header_clean) + 50].strip()
        
        # If header is found at start, remove it
        if content_start.startswith(header_clean):
            # Remove header and any following whitespace/newlines
            remaining = content[len(header_clean):].strip()
            # Remove leading colons, dashes, or whitespace
            remaining = re.sub(r'^[:–-\s]+', '', remaining)
            return remaining
        
        return content
    
    def _build_content_with_context(self, header: str, content: str, document_title: Optional[str]) -> str:
        """Build content with document title and header context."""
        parts = []
        if document_title:
            parts.append(f"**{document_title}**")
        if header:
            parts.append(f"\n{header}")
        parts.append(f"\n\n{content}")
        return "\n".join(parts)
    
    def _extract_metadata(self, chunk: DocumentChunk, header: str, content: str, document_title: Optional[str]):
        """
        Extract metadata using pattern-based techniques.
        No LLM required - uses regex patterns and scoring-based classification.
        ALL VALUES IN ARABIC.
        
        Separated into logical sections:
        1. Article extraction (Arabic format)
        2. Section/Chapter extraction (Arabic format)
        3. Numbered section extraction
        4. Classification
        5. Build resource path in Arabic
        """
        # ===== ARTICLE EXTRACTION (ARABIC) =====
        article_number = None
        article_number_arabic = None  # Store Arabic representation
        
        # Try header first
        if "المادة" in header:
            article_number = self._number_parser.parse_article_number(header)
            if article_number:
                # Extract Arabic form
                article_match = re.search(
                    r'(المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?))',
                    header
                )
                if article_match:
                    article_number_arabic = article_match.group(1).strip()
                else:
                    article_number_arabic = f"المادة {article_number}"
                
                chunk.article_reference = article_number_arabic  # NOW IN ARABIC
                chunk.item_type = "article"
                chunk.item_number = article_number
        
        # If not found in header, try content (for split chunks)
        if not article_number:
            # Look for article reference in content
            article_match = re.search(
                r'(المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?))',
                content[:500]
            )
            if article_match:
                article_number_arabic = article_match.group(1).strip()
                article_number = self._number_parser.parse_article_number(article_number_arabic)
                if article_number:
                    chunk.article_reference = article_number_arabic  # NOW IN ARABIC
                    chunk.item_type = "article"
                    chunk.item_number = article_number
        
        # Extract article title if we have article number
        if chunk.article_reference:
            title_match = re.search(
                r'المادة\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|\.|$)',
                content[:400]
            )
            if title_match:
                chunk.item_title = title_match.group(1).strip()
        
        # ===== PART/CHAPTER EXTRACTION (already in hierarchy context) =====
        # Already set from hierarchy context, but double-check from content
        if "الباب" in header or "الباب" in content[:300]:
            if not chunk.legal_part_name:
                title_match = re.search(
                    r'الباب\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    content[:300]
                )
                if title_match:
                    chunk.legal_part_name = title_match.group(1).strip()
        
        if "الفصل" in header or "الفصل" in content[:300]:
            if not chunk.legal_chapter_name:
                title_match = re.search(
                    r'الفصل\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    content[:300]
                )
                if title_match:
                    chunk.legal_chapter_name = title_match.group(1).strip()
        
        # ===== NUMBERED SECTION EXTRACTION =====
        numbered_match = re.match(r'^(\d+)[\.\)]\s+(.+?)(?=\n|$)', header)
        if numbered_match:
            chunk.item_number = numbered_match.group(1)
            chunk.item_type = "section"
            chunk.item_title = numbered_match.group(2).strip()
        
        # ===== CLASSIFICATION =====
        # Use scoring-based classification
        chunk.category = self._classifier.classify_category(content, document_title)
        chunk.target_audience = self._classifier.classify_target_audience(content, document_title)
        
        # ===== KEYWORD EXTRACTION =====
        # Extract keywords using KeyBERT (use chunk.content which includes full context)
        chunk.keywords = self._keyword_extractor.extract_keywords(chunk.content)
        
        # ===== BUILD RESOURCE PATH (ARABIC) =====
        chunk.metadata['resource_path'] = self._build_resource_path(chunk, document_title)
    
    def _build_resource_path(self, chunk: DocumentChunk, document_title: Optional[str]) -> str:
        """
        Build hierarchical resource path in Arabic for citations.
        Example: "نظام العمل > الباب الخامس > الفصل الأول > المادة 151"
        """
        parts = []
        
        # Add document title (or name)
        if document_title:
            parts.append(document_title)
        elif chunk.document_name:
            # Remove .pdf extension for cleaner display
            doc_name = chunk.document_name.replace('.pdf', '')
            parts.append(doc_name)
        
        # Add part (الباب)
        if chunk.legal_part_name:
            parts.append(chunk.legal_part_name)
        
        # Add chapter (الفصل)
        if chunk.legal_chapter_name:
            parts.append(chunk.legal_chapter_name)
        
        # Add article (المادة)
        if chunk.article_reference:
            parts.append(chunk.article_reference)
        elif chunk.item_type == "section" and chunk.item_number:
            parts.append(f"البند {chunk.item_number}")
        
        return " > ".join(parts)