import re
from typing import List, Optional
from collections import Counter
from core.utils.logger import logger

try:
    from keybert import KeyBERT  # pyright: ignore[reportMissingImports]
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    logger.warning("KeyBERT not available. Install with: pip install keybert sentence-transformers")


class KeywordExtractor:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", top_n: int = 5):
        # Uses multilingual model for Arabic support
        self.top_n = top_n
        self.model_name = model_name
        self.keybert = None
        
        if not KEYBERT_AVAILABLE:
            logger.warning("KeyBERT not available. Keyword extraction will use fallback method.")
            logger.warning("For better results, install: pip install keybert sentence-transformers")
            self.keybert = None
            return
        
        try:
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
        if not text or len(text.strip()) < 20:
            return []
        
        top_n = top_n or self.top_n
        
        if self.keybert is not None:
            try:
                return self._extract_with_keybert(text, top_n)
            except Exception as e:
                logger.warning(f"KeyBERT extraction failed: {str(e)}, using fallback")
        
        return self._extract_simple_keywords(text, top_n)
    
    def _extract_with_keybert(self, text: str, top_n: int) -> List[str]:
        clean_text = ' '.join(text.split())
        
        keywords_with_scores = self.keybert.extract_keywords(  # pyright: ignore[reportOptionalMemberAccess]
            clean_text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,  # Don't filter stop words (important for Arabic)
            top_n=top_n * 2,
            use_mmr=False,
            diversity=0.0
        )
        
        keywords = []
        for kw_tuple in keywords_with_scores:
            if isinstance(kw_tuple, tuple) and len(kw_tuple) >= 2:
                keyword, score = kw_tuple[0], kw_tuple[1]
                # Lower threshold for Arabic (0.05 instead of 0.1)
                if score > 0.05 and keyword and len(keyword.strip()) > 1:
                    keywords.append(keyword.strip())
            elif isinstance(kw_tuple, str):
                if len(kw_tuple.strip()) > 1:
                    keywords.append(kw_tuple.strip())
        
        return keywords[:top_n]
    
    def _extract_simple_keywords(self, text: str, top_n: int) -> List[str]:
        # Fallback: frequency-based extraction for Arabic words
        words = re.findall(r'[\u0600-\u06FF]{2,}', text)
        word_counts = Counter(words)
        
        keywords = []
        for word, count in word_counts.most_common(top_n * 3):
            if len(word) >= 2:
                keywords.append(word)
                if len(keywords) >= top_n:
                    break
        
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
        results = []
        for text in texts:
            keywords = self.extract_keywords(text, top_n)
            results.append(keywords)
        
        return results
