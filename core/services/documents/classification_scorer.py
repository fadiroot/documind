"""
Classification scorer for document chunks.

Uses scoring-based approach instead of naive keyword matching.
Supports weighted keywords and context-aware classification.
"""
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


class ClassificationScorer:
    """
    Scoring-based classifier for categories and target audiences.
    
    Uses weighted keyword matching with context awareness.
    Returns the highest-scoring category/audience.
    """
    
    # Category keywords with weights (higher = more specific/important)
    # Keys are now in Arabic for consistent metadata
    CATEGORY_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
        'الإجازات': [  # Leave
            ('إجازة مرضية', 3.0), ('إجازة أمومة', 3.0), ('إجازة زواج', 3.0),
            ('إجازة اعتيادية', 2.5), ('إجازة', 2.0), ('إجازات', 2.0),
            ('الإجازات', 2.5), ('طلب إجازة', 2.5),
        ],
        'الحقوق المالية': [  # Financial Rights
            ('حقوق مالية', 3.5), ('مستحقات', 3.0), ('راتب', 2.5),
            ('بدل', 2.5), ('مكافأة', 2.5), ('علاوة', 2.5),
            ('أجر', 2.0), ('أجور', 2.0), ('الراتب', 2.5),
            ('البدل', 2.5), ('المكافأة', 2.5),
        ],
        'الانضباط': [  # Discipline
            ('انضباط', 3.5), ('عقوبة', 3.0), ('جزاء', 3.0),
            ('تأديب', 3.0), ('مخالفة', 2.5), ('عقوبات', 2.5),
            ('الانضباط', 3.5), ('العقوبة', 3.0),
        ],
        'التوظيف': [  # Recruitment
            ('تعيين', 3.0), ('توظيف', 3.0), ('اختيار', 2.5),
            ('ترشيح', 2.5), ('قبول', 2.0), ('التعيين', 3.0),
            ('التوظيف', 3.0),
        ],
        'الترقية': [  # Promotion
            ('ترقية', 3.0), ('ترقيات', 3.0), ('ترقي', 2.5),
            ('الترقية', 3.0), ('الترقيات', 3.0),
        ],
        'الأداء': [  # Performance
            ('تقييم الأداء', 3.5), ('أداء', 2.5), ('تقييم', 2.5),
            ('كفاءة', 2.5), ('إنجاز', 2.0), ('الأداء', 2.5),
            ('التقييم', 2.5),
        ],
    }
    
    # Target audience keywords with weights
    # Keys are now in Arabic for consistent metadata
    AUDIENCE_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
        'المهندسون': [  # Engineers
            ('المهندسين', 3.0), ('المهندس', 2.5), ('مهندس', 2.0),
            ('هندسي', 2.5), ('هندسة', 2.0), ('الهندسة', 2.5),
        ],
        'الموظفون المدنيون': [  # General Civil Servants
            ('الموظفين', 3.0), ('الموظف', 2.5), ('موظف', 2.0),
            ('موظفين', 2.0), ('موظفي', 2.0), ('الموظفين', 3.0),
        ],
        'المتعاقدون': [  # Contractors
            ('المتعاقدين', 3.0), ('المتعاقد', 2.5), ('متعاقد', 2.0),
            ('متعاقدين', 2.0), ('عقد', 1.5), ('العقد', 1.5),
        ],
        'العمال': [  # Labourers
            ('العمال', 3.0), ('العامل', 2.5), ('عامل', 2.0),
            ('عمال', 2.0), ('أجير', 2.0), ('أجراء', 2.0),
            ('الأجير', 2.0), ('الأجراء', 2.0),
        ],
    }
    
    @classmethod
    def classify_category(
        cls,
        content: str,
        document_title: Optional[str] = None,
        min_score: float = 1.0
    ) -> Optional[str]:
        """
        Classify category using scoring-based approach.
        
        Args:
            content: Chunk content
            document_title: Document title (weighted higher)
            min_score: Minimum score threshold
        
        Returns:
            Category name or None
        """
        scores = defaultdict(float)
        
        # Normalize text
        content_lower = content.lower()
        title_lower = (document_title or "").lower()
        
        # Score each category
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            # Check title first (higher weight)
            for keyword, weight in keywords:
                if keyword in title_lower:
                    scores[category] += weight * 1.5  # Title boost
            
            # Check content
            for keyword, weight in keywords:
                count = content_lower.count(keyword)
                if count > 0:
                    scores[category] += weight * count
        
        # Return highest scoring category if above threshold
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] >= min_score:
                return best_category[0]
        
        return None
    
    @classmethod
    def classify_target_audience(
        cls,
        content: str,
        document_title: Optional[str] = None,
        min_score: float = 1.0
    ) -> Optional[str]:
        """
        Classify target audience using scoring-based approach.
        
        Args:
            content: Chunk content
            document_title: Document title (weighted higher)
            min_score: Minimum score threshold
        
        Returns:
            Audience name or None
        """
        scores = defaultdict(float)
        
        # Normalize text
        content_lower = content.lower()
        title_lower = (document_title or "").lower()
        
        # Score each audience
        for audience, keywords in cls.AUDIENCE_KEYWORDS.items():
            # Check title first (higher weight)
            for keyword, weight in keywords:
                if keyword in title_lower:
                    scores[audience] += weight * 2.0  # Title boost
            
            # Check content
            for keyword, weight in keywords:
                count = content_lower.count(keyword)
                if count > 0:
                    scores[audience] += weight * count
        
        # Return highest scoring audience if above threshold
        if scores:
            best_audience = max(scores.items(), key=lambda x: x[1])
            if best_audience[1] >= min_score:
                return best_audience[0]
        
        return None
