"""Logging utility."""
import logging
import sys
from typing import Optional


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Set up and return a logger instance."""
    logger = logging.getLogger(name)
    
    if level is None:
        level = logging.INFO
    
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# Default logger instance
logger = setup_logger(__name__)
