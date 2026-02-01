# Core Services Architecture

## Overview

The `core/services` package provides production-ready, domain-organized services for the DocuMind RAG system. All services are designed with clean architecture principles, proper error handling, and comprehensive documentation.

## Service Categories

### 1. **Agent Services** (`agents/`)
Main question-answering services using RAG (Retrieval-Augmented Generation).

#### `AgentService`
**Purpose**: Main entry point for question answering with conversation management.

**Key Features**:
- RAG-based question answering via LangChain agent
- Conversation memory with history passed to agent
- Streaming responses
- Optional Azure thread persistence

**Usage**:
```python
from core.services import AgentService

agent = AgentService(min_retrieval_score=0.3)
for event in agent.stream(
    question="ما هي أنواع الإجازات المتاحة؟",
    user=user_metadata,
    session_id="user_123"
):
    if event["type"] == "answer_chunk":
        print(event["content"], end="")
    elif event["type"] == "complete":
        break
```

**Internal Components**:
- `AgentChain`: Streams from LangChain agent, injects conversation history
- `agent.py`: `create_documind_agent()` – LangChain agent with `retrieve_tool`
- `azure_client.py`: Azure OpenAI LLM factory (`get_azure_llm`)
- `RetrievalService.create_retrieve_tool`: Document retrieval tool for the agent

---

### 2. **Document Services** (`documents/`)
PDF processing, text extraction, and intelligent chunking.

#### `PDFService`
**Purpose**: Extract and process text from PDF documents.

**Usage**:
```python
from core.services import PDFService

pdf_service = PDFService()
with open("document.pdf", "rb") as f:
    chunks = pdf_service.chunk_pdf_with_metadata(
        pdf_bytes=f.read(),
        filename="document.pdf"
    )

for chunk in chunks:
    print(chunk.content)
    print(chunk.metadata)
```

#### `DocumentChunker`
**Purpose**: Split documents into searchable chunks with rich metadata extraction.

**Features**:
- Hierarchical structure extraction (Part > Chapter > Article)
- Arabic number parsing
- Classification (category, target audience)
- Keyword extraction using KeyBERT

**Internal Components**:
- `arabic_number_parser.py`: Parse Arabic ordinal numbers
- `classification_scorer.py`: Rule-based document classification
- `keyword_extractor.py`: KeyBERT-based keyword extraction

---

### 3. **Indexing Services** (`indexing/`)
Azure AI Search index management and document upload.

#### `IndexService`
**Purpose**: Create and manage Azure AI Search index schema.

**Usage**:
```python
from core.services import IndexService

index_service = IndexService()

# Create or update index
index_service.create_index(vector_dimension=3072)

# Check if index exists
if index_service.index_exists():
    print("Index is ready")

# Delete index
index_service.delete_index()
```

#### `StorageService`
**Purpose**: Upload documents to Azure AI Search index.

**Usage**:
```python
from core.services import StorageService

storage = StorageService()

documents = [
    {
        "id": "doc_1",
        "content": "Document text...",
        "contentVector": [0.1, 0.2, ...],  # 3072-dim embedding
        "source_document": "document.pdf",
        "category": "الإجازات",
        # ... other fields
    }
]

success = storage.upload_documents(documents)
```

---

### 4. **Retrieval Services** (`retrieval/`) - Internal
Document search and embedding generation (internal use only).

#### `RetrievalService`
**Purpose**: Retrieve relevant documents with quality controls. Exposes `create_retrieve_tool()` for LangChain agent integration.

**Features**:
- Vector similarity search
- Score thresholding
- Context length management
- Automatic threshold relaxation
- `create_retrieve_tool(retrieval_service)`: Returns a LangChain `@tool` for document retrieval

#### `EmbeddingService`
**Purpose**: Generate OpenAI embeddings for text.

#### `SearchService`
**Purpose**: Azure AI Search client wrapper.

---

### 5. **Supporting Services**

#### Memory (`memory/`)
- `ConversationMemory`: In-memory history (sliding window) and optional Azure AI Agents thread persistence

#### Prompts (`prompts/`)
- `PromptBuilder`: Format context and build prompts for LLM
- `templates/system_prompt.promptly`: System prompt for LangChain agent (retrieval rules)

#### Errors (`errors/`)
- `ErrorHandler`: Centralized error handling
- `FallbackResponses`: Fallback messages in Arabic/English

#### Auth (`auth/`)
- `auth_service`: User authentication and authorization

#### Utils (`utils/`)
- `metadata_utils.py`: Shared metadata processing functions

---

## Architecture Principles

### 1. **Clean Separation of Concerns**
Each service has a single, well-defined responsibility:
- `AgentService`: Question answering orchestration
- `PDFService`: Document processing
- `IndexService`: Index management
- `RetrievalService`: Document search

### 2. **Dependency Injection**
Services accept dependencies in constructors for testability:
```python
agent = AgentService(min_retrieval_score=0.3)
retrieval = RetrievalService(min_score_threshold=0.3, enable_reranking=False)
```

### 3. **Type Safety**
All services use comprehensive type hints:
```python
def answer_question(
    self,
    question: str,
    user_metadata: Optional[UserMetadata] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    ...
```

### 4. **Error Handling**
All services include:
- Try-except blocks for external services
- Detailed error logging
- Graceful degradation
- User-friendly error messages

### 5. **Documentation**
Every service includes:
- Comprehensive docstrings
- Usage examples
- Parameter descriptions
- Return value specifications

---

## Service Dependencies

```
AgentService
├── AgentChain
│   ├── create_documind_agent (agent.py)
│   │   ├── get_azure_llm (azure_client)
│   │   ├── RetrievalService
│   │   │   ├── EmbeddingService (OpenAI)
│   │   │   └── SearchService (Azure AI Search)
│   │   └── create_retrieve_tool (retrieve_tool)
│   └── ConversationMemory (history injected into agent)
│
PDFService
├── DocumentChunker
│   ├── ArabicNumberParser
│   ├── ClassificationScorer
│   └── KeywordExtractor
└── PyMuPDF

IndexService
└── Azure SDK

StorageService
└── Azure SDK
```

---

## Configuration

All services use centralized configuration from `app/config.py`:

```python
# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT = "https://..."
AZURE_AI_SEARCH_API_KEY = "..."
AZURE_AI_SEARCH_INDEX_NAME = "documents-index"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = "https://..."
AZURE_OPENAI_API_KEY = "..."
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

# Azure AI Agents (optional, for thread persistence)
AZURE_PROJECT_ENDPOINT = "https://..."
AZURE_AI_AGENT_ID = "..."
```

---

## Testing

### Unit Tests
Each service should have unit tests:
```python
def test_agent_service_stream():
    agent = AgentService()
    events = list(agent.stream("test question", session_id="test_123"))
    answer_chunks = [e["content"] for e in events if e.get("type") == "answer_chunk"]
    assert any(answer_chunks) or any(e.get("type") == "complete" for e in events)
```

### Integration Tests
Test service interactions:
```python
def test_end_to_end_pipeline():
    # 1. Process PDF
    pdf_service = PDFService()
    chunks = pdf_service.chunk_pdf_with_metadata(pdf_bytes)
    
    # 2. Create embeddings and index
    embedding_service = EmbeddingService()
    embeddings = embedding_service.create_embeddings([c.content for c in chunks])
    
    # 3. Upload to index
    storage = StorageService()
    storage.upload_documents(documents)
    
    # 4. Query
    agent = AgentService()
    events = list(agent.stream("test", session_id="test_123"))
    assert any(e.get("type") == "complete" or e.get("type") == "answer_chunk" for e in events)
```

---

## Best Practices

### 1. **Always Use Type Hints**
```python
def process_document(filename: str, content: bytes) -> List[DocumentChunk]:
    ...
```

### 2. **Log Important Operations**
```python
logger.info(f"Processing document: {filename}")
logger.error(f"Failed to process: {str(e)}")
```

### 3. **Handle Errors Gracefully**
```python
try:
    result = external_service.call()
except ServiceError as e:
    logger.error(f"Service error: {str(e)}")
    return fallback_response()
```

### 4. **Use Descriptive Names**
- Services: `AgentService`, `PDFService`
- Methods: `answer_question()`, `chunk_pdf_with_metadata()`
- Variables: `min_retrieval_score`, `document_chunks`

### 5. **Document Public APIs**
All public methods must have:
- Docstring with description
- Args section
- Returns section
- Usage example

---

## Production Checklist

- [x] Clean code organization
- [x] Comprehensive docstrings
- [x] Type hints everywhere
- [x] Error handling
- [x] Logging
- [x] No unused imports
- [x] No duplicate code
- [x] Shared utilities extracted
- [x] Clean `__init__.py` exports
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance monitoring
- [ ] Rate limiting
- [ ] Caching layer

---

## Migration Guide

### From Old to New API

**Old** (deprecated):
```python
from core.services import AzureAISearchRetriever, AgentChain
# Direct chain usage (complex)
```

**New** (recommended):
```python
from core.services import AgentService
agent = AgentService()
for event in agent.stream(question=question, session_id=session_id):
    if event["type"] == "answer_chunk":
        print(event["content"], end="")
```

---

## Support

For issues or questions:
1. Check this documentation
2. Review service docstrings
3. Check logs for error details
4. Contact the development team

---

**Last Updated**: February 2026  
**Version**: 2.1 (LangChain agent with conversation history)
