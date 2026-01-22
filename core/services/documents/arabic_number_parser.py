"""
Arabic number parser utility for legal document processing.

Handles conversion of Arabic ordinal numbers (written and numeric) to integers.
Supports numbers from 1 to 99+ using rule-based parsing.
"""
import re
from typing import Optional


class ArabicNumberParser:
    """
    Rule-based parser for Arabic ordinal numbers.
    
    Supports:
    - Numeric: "المادة 37" -> 37
    - Simple ordinals (1-10): "المادة الأولى" -> 1
    - Compound ordinals (11-99): "المادة السابعة والثلاثون" -> 37
    """
    
    # Units (1-10) - both masculine and feminine forms
    UNITS = {
        'الأول': 1, 'الأولى': 1,
        'الثاني': 2, 'الثانية': 2,
        'الثالث': 3, 'الثالثة': 3,
        'الرابع': 4, 'الرابعة': 4,
        'الخامس': 5, 'الخامسة': 5,
        'السادس': 6, 'السادسة': 6,
        'السابع': 7, 'السابعة': 7,
        'الثامن': 8, 'الثامنة': 8,
        'التاسع': 9, 'التاسعة': 9,
        'العاشر': 10, 'العاشرة': 10,
    }
    
    # Tens (10, 20, 30, ..., 90)
    TENS = {
        'عشر': 10, 'عشرة': 10,
        'عشرون': 20, 'عشرين': 20,
        'ثلاثون': 30, 'ثلاثين': 30,
        'أربعون': 40, 'أربعين': 40,
        'خمسون': 50, 'خمسين': 50,
        'ستون': 60, 'ستين': 60,
        'سبعون': 70, 'سبعين': 70,
        'ثمانون': 80, 'ثمانين': 80,
        'تسعون': 90, 'تسعين': 90,
    }
    
    # Special case for 11-19 (compound with "عشر")
    COMPOUND_TENS_PREFIXES = {
        'أول': 1, 'أولى': 1,
        'ثاني': 2, 'ثانية': 2,
        'ثالث': 3, 'ثالثة': 3,
        'رابع': 4, 'رابعة': 4,
        'خامس': 5, 'خامسة': 5,
        'سادس': 6, 'سادسة': 6,
        'سابع': 7, 'سابعة': 7,
        'ثامن': 8, 'ثامنة': 8,
        'تاسع': 9, 'تاسعة': 9,
    }
    
    @classmethod
    def parse_article_number(cls, text: str) -> Optional[str]:
        """
        Extract and parse article number from text.
        
        Args:
            text: Text containing article reference (e.g., "المادة 37" or "المادة السابعة والثلاثون")
        
        Returns:
            Article number as string, or None if not found
        """
        # First try numeric extraction
        numeric_match = re.search(r'المادة\s+(\d+)', text)
        if numeric_match:
            return numeric_match.group(1)
        
        # Try to find Arabic ordinal pattern
        ordinal_match = re.search(
            r'المادة\s+([^\s]+(?:\s+و\s+[^\s]+)?)',
            text
        )
        if not ordinal_match:
            return None
        
        ordinal_text = ordinal_match.group(1).strip()
        number = cls._parse_ordinal(ordinal_text)
        
        return str(number) if number is not None else None
    
    @classmethod
    def _parse_ordinal(cls, text: str) -> Optional[int]:
        """
        Parse Arabic ordinal text to integer.
        
        Args:
            text: Arabic ordinal text (e.g., "السابعة والثلاثون")
        
        Returns:
            Integer value or None
        """
        text = text.strip()
        
        # Check simple units (1-10)
        for ordinal, value in cls.UNITS.items():
            if ordinal in text:
                # If it's just a simple unit, return it
                if text == ordinal or text.startswith(ordinal):
                    return value
        
        # Check for compound numbers (11-99)
        # Pattern: "السابعة والثلاثون" = 37 (7 + 30)
        # Pattern: "الحادي عشر" = 11
        
        # Check for 11-19 pattern: "الحادي عشر", "الثاني عشر", etc.
        for prefix, unit_value in cls.COMPOUND_TENS_PREFIXES.items():
            if prefix in text and ('عشر' in text or 'عشرة' in text):
                # Check if it's 11-19
                if 'عشر' in text or 'عشرة' in text:
                    # Verify it's not 20+ (which would have "عشرون" or "عشرين")
                    if 'عشرون' not in text and 'عشرين' not in text:
                        return 10 + unit_value
        
        # Check for compound pattern: "X والY" where X is unit and Y is tens
        # Example: "السابعة والثلاثون" = 7 + 30 = 37
        if 'و' in text or 'وال' in text:
            parts = re.split(r'\s+و\s*ال?', text)
            if len(parts) == 2:
                unit_part = parts[0].strip()
                tens_part = parts[1].strip()
                
                # Find unit value
                unit_value = None
                for ordinal, value in cls.UNITS.items():
                    if ordinal in unit_part:
                        unit_value = value
                        break
                
                # Find tens value
                tens_value = None
                for tens_word, value in cls.TENS.items():
                    if tens_word in tens_part:
                        tens_value = value
                        break
                
                if unit_value is not None and tens_value is not None:
                    return unit_value + tens_value
        
        # Check for standalone tens (20, 30, 40, etc.)
        for tens_word, value in cls.TENS.items():
            if tens_word in text:
                # Make sure it's not part of a compound
                if 'و' not in text and 'وال' not in text:
                    return value
        
        return None
    
    @classmethod
    def extract_number_from_text(cls, text: str, max_length: int = 500) -> Optional[str]:
        """
        Extract any Arabic number (article, section, etc.) from text.
        
        Args:
            text: Text to search
            max_length: Maximum length of text to search (for performance)
        
        Returns:
            Number as string, or None
        """
        # Try numeric first
        numeric_match = re.search(r'(\d+)', text[:max_length])
        if numeric_match:
            return numeric_match.group(1)
        
        # Try Arabic ordinal
        ordinal_match = re.search(
            r'([^\s]+(?:\s+و\s+[^\s]+)?)',
            text[:max_length]
        )
        if ordinal_match:
            ordinal_text = ordinal_match.group(1).strip()
            number = cls._parse_ordinal(ordinal_text)
            if number is not None:
                return str(number)
        
        return None
