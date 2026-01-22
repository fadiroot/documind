"""Keyword extraction service using KeyBERT."""
import re
from typing import List, Optional
from collections import Counter
from core.utils.logger import logger

try:
    from keybert import KeyBERT # type: ignore
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    logger.warning("KeyBERT not available. Install with: pip install keybert sentence-transformers")


class KeywordExtractor:
    """
    Keyword extraction service using KeyBERT.
    
    KeyBERT uses BERT embeddings to find keywords that are most similar to the document.
    Works well with Arabic text when using multilingual models.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", top_n: int = 5):
        """
        Initialize keyword extractor.
        
        Args:
            model_name: Sentence transformer model name (default: multilingual model for Arabic support)
            top_n: Number of keywords to extract per document (default: 5)
        """
        self.top_n = top_n
        self.model_name = model_name
        self.keybert = None
        
        if not KEYBERT_AVAILABLE:
            logger.warning("KeyBERT not available. Keyword extraction will use fallback method.")
            logger.warning("For better results, install: pip install keybert sentence-transformers")
            self.keybert = None
            return
        
        try:
            # Use multilingual model for Arabic support
            logger.info(f"Initializing KeyBERT with model: {model_name}")
            self.keybert = KeyBERT(model=model_name)
            logger.info("KeyBERT initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KeyBERT: {str(e)}")
            logger.warning("Falling back to simple keyword extraction method")
            import traceback
            logger.debug(traceback.format_exc())
            self.keybert = None
    
    def extract_keywords(self, text: str, top_n: Optional[int] = None) -> List[str]:
        """
        Extract keywords from text using KeyBERT or fallback method.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of keywords to extract (defaults to instance top_n)
        
        Returns:
            List of keyword strings (empty list if extraction fails)
        """
        if not text or len(text.strip()) < 20:
            return []
        
        top_n = top_n or self.top_n
        
        # Use KeyBERT if available
        if self.keybert is not None:
            try:
                return self._extract_with_keybert(text, top_n)
            except Exception as e:
                logger.warning(f"KeyBERT extraction failed: {str(e)}, using fallback")
        
        # Fallback to simple frequency-based extraction
        return self._extract_simple_keywords(text, top_n)
    
    def _extract_with_keybert(self, text: str, top_n: int) -> List[str]:
        """Extract keywords using KeyBERT."""
        # Clean text - remove extra whitespace and newlines
        clean_text = ' '.join(text.split())
        
        # Extract keywords with scores - try simpler approach first
        keywords_with_scores = self.keybert.extract_keywords( # type: ignore
            clean_text,
            keyphrase_ngram_range=(1, 2),  # Extract 1-word and 2-word phrases
            stop_words=None,  # Don't filter stop words (important for Arabic)
            top_n=top_n * 2,  # Get more candidates
            use_mmr=False,  # Disable MMR for more reliable results
            diversity=0.0  # No diversity filtering
        )
        
        # Extract keywords - lower threshold for Arabic text
        keywords = []
        for kw_tuple in keywords_with_scores:
            if isinstance(kw_tuple, tuple) and len(kw_tuple) >= 2:
                keyword, score = kw_tuple[0], kw_tuple[1]
                # Lower threshold for Arabic (0.05 instead of 0.1)
                if score > 0.05 and keyword and len(keyword.strip()) > 1:
                    keywords.append(keyword.strip())
            elif isinstance(kw_tuple, str):
                # Sometimes KeyBERT returns just strings
                if len(kw_tuple.strip()) > 1:
                    keywords.append(kw_tuple.strip())
        
        return keywords[:top_n]
    
    def _extract_simple_keywords(self, text: str, top_n: int) -> List[str]:
        """
        Simple frequency-based keyword extraction fallback.
        Extracts most common Arabic words/phrases.
        """
        # Extract Arabic words (2+ characters)
        words = re.findall(r'[\u0600-\u06FF]{2,}', text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get top keywords
        keywords = []
        for word, count in word_counts.most_common(top_n * 3):
            if len(word) >= 2:
                keywords.append(word)
                if len(keywords) >= top_n:
                    break
        
        # Also try 2-word phrases
        if len(keywords) < top_n:
            phrases = []
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) >= 4:
                    phrases.append(phrase)
            
            phrase_counts = Counter(phrases)
            for phrase, count in phrase_counts.most_common(top_n - len(keywords)):
                if phrase not in keywords:
                    keywords.append(phrase)
                    if len(keywords) >= top_n:
                        break
        
        return keywords[:top_n]
    
    def extract_keywords_batch(self, texts: List[str], top_n: Optional[int] = None) -> List[List[str]]:
        """
        Extract keywords from multiple texts.
        
        Args:
            texts: List of texts to extract keywords from
            top_n: Number of keywords per text
        
        Returns:
            List of keyword lists (one per input text)
        """
        results = []
        for text in texts:
            keywords = self.extract_keywords(text, top_n)
            results.append(keywords)
        
        return results
