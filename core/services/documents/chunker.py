import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from core.utils.logger import logger
from core.services.documents.arabic_number_parser import ArabicNumberParser
from core.services.documents.classification_scorer import ClassificationScorer
from core.services.documents.keyword_extractor import KeywordExtractor


@dataclass
class DocumentChunk:
    content: str
    document_name: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    item_number: Optional[str] = None
    item_type: Optional[str] = None
    article_reference: Optional[str] = None
    section_title: Optional[str] = None
    item_title: Optional[str] = None
    legal_part_name: Optional[str] = None
    legal_chapter_name: Optional[str] = None
    category: Optional[str] = None
    target_audience: Optional[str] = None
    keywords: Optional[List[str]] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    chunk_index: Optional[int] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None


@dataclass
class HierarchyContext:
    document_title: Optional[str] = None
    current_part: Optional[str] = None
    current_part_number: Optional[str] = None
    current_chapter: Optional[str] = None
    current_chapter_number: Optional[str] = None
    current_article_number: Optional[str] = None


class DocumentChunker:
    # Patterns match headers at start of line (using ^ with MULTILINE flag)
    HEADER_PATTERNS = [
        r'^\s*الباب\s+',
        r'^\s*الفصل\s+',
        r'^\s*المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?)',
        r'^\s*\d+[\.\)]\s+',
    ]
    
    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 200):
        # Cap at 2000 for safety (embedding models typically have 8K token limit)
        self.max_chunk_size = min(max_chunk_size, 2000)
        self.chunk_overlap = chunk_overlap
        self._number_parser = ArabicNumberParser()
        self._classifier = ClassificationScorer()
        self._keyword_extractor = KeywordExtractor(top_n=5)
        self._hierarchy_context = HierarchyContext()
    
    def chunk_document(self, text: str, source_file: str) -> List[DocumentChunk]:
        text = self._normalize_text(text)
        document_title = self._extract_title(text)
        self._hierarchy_context = HierarchyContext(document_title=document_title)
        blocks = self._split_by_headers(text)
        chunks = []
        chunk_idx = 0
        for header, content in blocks:
            self._update_hierarchy_context(header, content)
            content = self._remove_header_duplication(header, content)
            if len(content) > self.max_chunk_size:
                sub_blocks = self._split_large_block(content)
                for sub_content in sub_blocks:
                    full_content = self._build_content_with_context(header, sub_content, document_title)
                    chunk = self._create_chunk(
                        full_content, source_file, document_title, 
                        header, sub_content, chunk_idx
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                full_content = self._build_content_with_context(header, content, document_title)
                chunk = self._create_chunk(
                    full_content, source_file, document_title,
                    header, content, chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1
        return chunks
    
    def _update_hierarchy_context(self, header: str, content: str):
        if "الباب" in header:
            part_match = re.search(
                r'(الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))',
                header
            )
            if part_match:
                part_ref = part_match.group(1)
                title_match = re.search(
                    r'الباب\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    header + "\n" + content[:200]
                )
                if title_match:
                    part_title = title_match.group(1).strip()
                    self._hierarchy_context.current_part = f"{part_ref}: {part_title}"
                else:
                    self._hierarchy_context.current_part = part_ref
                # Reset chapter when entering new part
                self._hierarchy_context.current_chapter = None
                self._hierarchy_context.current_chapter_number = None
        if "الفصل" in header:
            chapter_match = re.search(
                r'(الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+))',
                header
            )
            if chapter_match:
                chapter_ref = chapter_match.group(1)
                title_match = re.search(
                    r'الفصل\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|$)',
                    header + "\n" + content[:200]
                )
                if title_match:
                    chapter_title = title_match.group(1).strip()
                    self._hierarchy_context.current_chapter = f"{chapter_ref}: {chapter_title}"
                else:
                    self._hierarchy_context.current_chapter = chapter_ref
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
        chunk = DocumentChunk(
            content=full_content.strip(),
            document_name=source_file,
            document_title=document_title,
            chunk_index=chunk_idx
        )
        chunk.legal_part_name = self._hierarchy_context.current_part
        chunk.legal_chapter_name = self._hierarchy_context.current_chapter
        self._extract_metadata(chunk, header, content, document_title)
        return chunk
    
    def _split_large_block(self, content: str) -> List[str]:
        # Try splitting by paragraphs first, then sentences
        paragraphs = re.split(r'\n\s*\n+', content)
        if len(paragraphs) > 1:
            sub_blocks = []
            current_block = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current_block and len(current_block) + len(para) + 2 > self.max_chunk_size:
                    sub_blocks.append(current_block)
                    # Add overlap between chunks
                    overlap = current_block[-self.chunk_overlap:] if len(current_block) > self.chunk_overlap else current_block
                    current_block = overlap + "\n\n" + para
                else:
                    current_block += "\n\n" + para if current_block else para
            if current_block:
                sub_blocks.append(current_block)
            final_blocks = []
            for block in sub_blocks:
                if len(block) > self.max_chunk_size:
                    final_blocks.extend(self._split_by_sentences(block))
                else:
                    final_blocks.append(block)
            return final_blocks if final_blocks else [content]
        else:
            return self._split_by_sentences(content)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'([.!?]\s+)', text)
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])
        if len(sentences) % 2 == 1:
            combined_sentences.append(sentences[-1])
        chunks = []
        current_chunk = ""
        for sentence in combined_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                chunks.append(current_chunk)
                # Add overlap between chunks
                overlap_sentences = current_chunk.split('.')[-3:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks if chunks else [text]
    
    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_title(self, text: str) -> Optional[str]:
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['لائحة', 'نظام', 'دليل', 'تعليمات', 'قرار']):
                if 20 <= len(line) <= 200:
                    return line
        return None
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        blocks = []
        header_positions = []
        for pattern in self.HEADER_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if self._is_structural_header(match, text):
                    header_positions.append((match.start(), match.group(0).strip()))
        header_positions = list(dict.fromkeys(header_positions))
        header_positions.sort(key=lambda x: x[0])
        if not header_positions:
            # No headers found, split by paragraphs to avoid huge chunks
            if len(text) > self.max_chunk_size:
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
        for i, (pos, header) in enumerate(header_positions):
            start = pos
            end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(text)
            content = text[start:end].strip()
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
        # Validate that match is at start of line, not in middle of sentence
        start_pos = match.start()
        if start_pos > 0:
            char_before = text[start_pos - 1]
            if char_before not in ['\n', '\r']:
                return False
        end_pos = match.end()
        if end_pos >= len(text):
            return False
        text_before = text[max(0, start_pos - 50):start_pos]
        if re.search(r'[.!?]\s*$', text_before):
            pass
        return True
    
    def _remove_header_duplication(self, header: str, content: str) -> str:
        # Prevents header from appearing both in header field and content
        if not header:
            return content
        header_clean = header.strip()
        content_start = content[:len(header_clean) + 50].strip()
        if content_start.startswith(header_clean):
            remaining = content[len(header_clean):].strip()
            remaining = re.sub(r'^[:–-\s]+', '', remaining)
            return remaining
        return content
    
    def _build_content_with_context(self, header: str, content: str, document_title: Optional[str]) -> str:
        parts = []
        if document_title:
            parts.append(f"**{document_title}**")
        if header:
            parts.append(f"\n{header}")
        parts.append(f"\n\n{content}")
        return "\n".join(parts)
    
    def _extract_metadata(self, chunk: DocumentChunk, header: str, content: str, document_title: Optional[str]):
        # Extract article references (Arabic format)
        article_number = None
        article_number_arabic = None
        if "المادة" in header:
            article_number = self._number_parser.parse_article_number(header)
            if article_number:
                article_match = re.search(
                    r'(المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?))',
                    header
                )
                if article_match:
                    article_number_arabic = article_match.group(1).strip()
                else:
                    article_number_arabic = f"المادة {article_number}"
                chunk.article_reference = article_number_arabic
                chunk.item_type = "article"
                chunk.item_number = article_number
        # If not found in header, try content (for split chunks)
        if not article_number:
            article_match = re.search(
                r'(المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+|(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s+و\s*ال(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\s*عشر)?))',
                content[:500]
            )
            if article_match:
                article_number_arabic = article_match.group(1).strip()
                article_number = self._number_parser.parse_article_number(article_number_arabic)
                if article_number:
                    chunk.article_reference = article_number_arabic
                    chunk.item_type = "article"
                    chunk.item_number = article_number
        if chunk.article_reference:
            title_match = re.search(
                r'المادة\s+[^\n:]+[:–-]?\s*(.+?)(?=\n|\.|$)',
                content[:400]
            )
            if title_match:
                chunk.item_title = title_match.group(1).strip()
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
        numbered_match = re.match(r'^(\d+)[\.\)]\s+(.+?)(?=\n|$)', header)
        if numbered_match:
            chunk.item_number = numbered_match.group(1)
            chunk.item_type = "section"
            chunk.item_title = numbered_match.group(2).strip()
        # Classification and keyword extraction
        chunk.category = self._classifier.classify_category(content, document_title)
        chunk.target_audience = self._classifier.classify_target_audience(content, document_title)
        chunk.keywords = self._keyword_extractor.extract_keywords(chunk.content)
        chunk.metadata['resource_path'] = self._build_resource_path(chunk, document_title)
    
    def _build_resource_path(self, chunk: DocumentChunk, document_title: Optional[str]) -> str:
        # Build hierarchical resource path in Arabic for citations
        # Example: "نظام العمل > الباب الخامس > الفصل الأول > المادة 151"
        parts = []
        if document_title:
            parts.append(document_title)
        elif chunk.document_name:
            doc_name = chunk.document_name.replace('.pdf', '')
            parts.append(doc_name)
        if chunk.legal_part_name:
            parts.append(chunk.legal_part_name)
        if chunk.legal_chapter_name:
            parts.append(chunk.legal_chapter_name)
        if chunk.article_reference:
            parts.append(chunk.article_reference)
        elif chunk.item_type == "section" and chunk.item_number:
            parts.append(f"البند {chunk.item_number}")
        return " > ".join(parts)
