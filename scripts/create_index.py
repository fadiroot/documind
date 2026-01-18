#!/usr/bin/env python3
"""Script to create or update the Azure AI Search index."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.services.documents.index_service import IndexService


def main():
    """Create or update the search index."""
    # Default vector dimension for text-embedding-3-large
    vector_dimension = 3072
    
    # Allow override via command line argument
    if len(sys.argv) > 1:
        try:
            vector_dimension = int(sys.argv[1])
        except ValueError:
            print(f"Invalid vector dimension: {sys.argv[1]}")
            print("Usage: python create_index.py [vector_dimension]")
            sys.exit(1)
    
    print(f"Creating index with vector dimension: {vector_dimension}")
    
    index_service = IndexService()
    
    if index_service.index_exists():
        print(f"Index '{index_service.index_name}' already exists. Updating...")
    else:
        print(f"Creating new index '{index_service.index_name}'...")
    
    success = index_service.create_index(vector_dimension=vector_dimension)
    
    if success:
        print("✓ Index created/updated successfully!")
        
        # Print index info
        index = index_service.get_index()
        if index:
            print(f"\nIndex details:")
            print(f"  Name: {index.name}")
            print(f"  Fields: {len(index.fields)}")
            for field in index.fields:
                print(f"    - {field.name} ({field.type})")
    else:
        print("✗ Failed to create/update index")
        sys.exit(1)


if __name__ == "__main__":
    main()
