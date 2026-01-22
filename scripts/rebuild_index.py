"""Script to rebuild the search index."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.azure_utils import get_search_client
from core.utils.logger import logger
from app.config import settings


def rebuild_index():
    """Rebuild the search index."""
    logger.info("Starting index rebuild...")
    logger.info(f"Target index: {settings.AZURE_SEARCH_INDEX_NAME}")
    
    from core.services.indexing.index_service import IndexService
    index_service = IndexService()
    
    if index_service.index_exists():
        logger.info("Index exists. Use create_index.py to update schema.")
    else:
        logger.info("Index does not exist. Use create_index.py to create it.")
    
    logger.info("Index rebuild check completed")


if __name__ == "__main__":
    rebuild_index()
