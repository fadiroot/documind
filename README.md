# DocuMind - Intelligent Document Q&A System

A production-ready Retrieval Augmented Generation (RAG) application built with FastAPI, Azure AI Services, and LangChain. DocuMind enables intelligent question-answering over document collections with conversation memory, user context awareness, and multilingual support (Arabic/English).

## 🚀 Features

### Core Capabilities
- **Document Processing**: PDF extraction and intelligent chunking
- **Vector Search**: Azure AI Search integration for semantic document retrieval
- **Intelligent Q&A**: LangChain-powered agents with tool calling
- **Conversation Memory**: Azure AI Agents threads for persistent conversation history
- **User Context**: Personalized responses based on user metadata (rank, cadre, etc.)
- **Multilingual Support**: Automatic Arabic/English language detection and response
- **Streaming Responses**: Real-time answer streaming via Server-Sent Events
- **Authentication**: JWT-based user authentication and authorization

### Advanced Features
- **Context-Aware Answers**: Uses user information (cadre, rank, position) for personalized responses
- **Session Management**: Conversation continuity across multiple interactions
- **Source Citation**: Returns document sources with confidence scores
- **Category Filtering**: Optional category-based question routing
- **Batch Processing**: Scripts for bulk document ingestion

## 📁 Project Structure

```
DocuMind/
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   ├── config.py                 # Configuration management
│   └── routes/                   # API route handlers
│       ├── auth.py               # Authentication endpoints
│       ├── docs.py                # Document upload/management
│       └── qa.py                  # Question-answering endpoints
│
├── core/                         # Core business logic
│   ├── models/                   # Pydantic data models
│   │   ├── document.py
│   │   ├── question.py
│   │   ├── response.py
│   │   └── user.py
│   │
│   ├── services/                 # Service layer (organized by domain)
│   │   ├── agents/               # Agent-related services
│   │   │   ├── agent_service.py      # Main agent orchestration
│   │   │   ├── agent_chain.py         # LangChain chain setup
│   │   │   ├── agent_tools.py         # Agent tools (user info)
│   │   │   ├── azure_retriever.py     # Azure AI Search retriever
│   │   │   └── conversation_memory.py # Azure thread management
│   │   │
│   │   ├── retrieval/            # Search & retrieval services
│   │   │   ├── retrieval_service.py   # Main retrieval orchestration
│   │   │   ├── embedding_service.py   # Text embedding generation
│   │   │   └── vectorstore_service.py  # Azure AI Search operations
│   │   │
│   │   ├── documents/            # Document processing
│   │   │   ├── pdf_service.py        # PDF extraction & chunking
│   │   │   └── index_service.py      # Index management
│   │   │
│   │   └── auth/                # Authentication
│   │       └── auth_service.py       # JWT & user management
│   │
│   └── utils/                    # Utility functions
│       ├── azure_utils.py
│       ├── logger.py
│       └── text_utils.py
│
├── scripts/                      # Utility scripts
│   ├── batch_process_documents.py  # Batch PDF processing
│   ├── create_index.py             # Create/update search index
│   ├── ingest_docs.py              # Single document ingestion
│   └── rebuild_index.py            # Index rebuild utility
│
├── documents/                    # Sample documents (PDFs)
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Prerequisites

- **Python**: 3.11 or higher
- **Docker & Docker Compose** (for containerized deployment)
- **Azure Services**:
  - Azure OpenAI (for LLM and embeddings)
  - Azure AI Search (for vector storage)
  - Azure AI Projects (for conversation memory - optional)
  - Azure Document Intelligence (optional, for advanced PDF processing)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Configuration
API_TITLE=DocuMind
API_VERSION=1.0.0
DEBUG=false

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_AI_SEARCH_API_KEY=your-search-key
AZURE_AI_SEARCH_INDEX_NAME=documents-index

# Azure AI Agents (for conversation memory)
AZURE_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
AZURE_AI_AGENT_ID=asst_xxxxxxxxxxxxx

# Azure Document Intelligence (optional)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Database (for user management)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
```

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DocuMind
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Database Admin: http://localhost:8080

### Local Development

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

## 📚 Usage Examples

### Authentication

```bash
# Login
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "password123"
  }'

# Response includes access_token
```

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/docs/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Ask Question (with conversation memory)

```bash
curl -X POST "http://localhost:8000/api/qa/ask?session_id=user123" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي سياسة الإجازات؟",
    "category": "legal"
  }'
```

### Streaming Response

```bash
curl -X POST "http://localhost:8000/api/qa/ask/stream?session_id=user123" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the benefits for my rank?",
    "category": "financial"
  }'
```

## 🔧 Scripts

### Create/Update Search Index

```bash
python scripts/create_index.py [vector_dimension]
# Default vector dimension: 3072 (for text-embedding-3-large)
```

### Batch Process Documents

```bash
python scripts/batch_process_documents.py /path/to/documents \
  --recursive \
  --pattern "*.pdf"
```

### Ingest Single Document

```bash
python scripts/ingest_docs.py /path/to/document.pdf
```

## 📡 API Endpoints

### Authentication (`/api/auth`)

- `POST /login` - Authenticate and get JWT token
- `GET /me` - Get current user information
- `GET /health` - Health check

### Documents (`/api/docs`)

- `POST /upload` - Upload and process PDF document
- `DELETE /{document_id}` - Delete document from index
- `GET /health` - Health check

### Q&A (`/api/qa`)

- `POST /ask` - Ask a question (supports `session_id` query param for conversation memory)
- `POST /ask/stream` - Stream answer tokens (SSE format)
- `GET /health` - Health check

### Query Parameters

- `session_id` (optional): Enables conversation memory across requests

## 🏗️ Architecture

### Service Organization

Services are organized by domain for better maintainability:

- **agents/**: Agent orchestration, LangChain chains, conversation memory
- **retrieval/**: Embedding generation, vector search, document retrieval
- **documents/**: PDF processing, text chunking, index management
- **auth/**: User authentication and JWT handling

### Key Components

1. **AgentService**: Main orchestration layer
   - Coordinates retrieval, LLM, and tools
   - Manages conversation memory
   - Handles user context

2. **AgentChain**: LangChain RAG chain
   - Processes questions with tools
   - Manages prompt engineering
   - Handles language detection

3. **ConversationMemory**: Azure AI Agents integration
   - Creates/manages conversation threads
   - Retrieves chat history
   - Saves messages to Azure threads

4. **RetrievalService**: Document search coordination
   - Creates query embeddings
   - Searches vector store
   - Formats results

## 🔐 Security

- JWT-based authentication
- Password hashing with bcrypt
- Secure credential management via environment variables
- CORS configuration for API access

## 🌐 Multilingual Support

- Automatic language detection (Arabic/English)
- Language-matched responses
- Supports Arabic documents and questions
- RTL text handling

## 📦 Dependencies

Key dependencies:
- `fastapi` - Web framework
- `langchain` - LLM orchestration
- `langchain-openai` - Azure OpenAI integration
- `azure-ai-projects` - Azure AI Agents
- `azure-search-documents` - Azure AI Search
- `azure-ai-documentintelligence` - PDF processing
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t documind:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  --env-file .env \
  documind:latest
```

### Docker Compose

```bash
docker-compose up -d
```

Includes:
- FastAPI application
- PostgreSQL database
- Adminer (database UI)

## 📝 Development Guidelines

### Code Organization

- Services organized by domain in `core/services/`
- Clear separation of concerns
- Type hints throughout
- Comprehensive docstrings

### Adding New Features

1. **New Service**: Add to appropriate domain subdirectory
2. **New Endpoint**: Add route handler in `app/routes/`
3. **New Model**: Add Pydantic model in `core/models/`
4. **Update Exports**: Update `__init__.py` files

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Azure Authentication**: Verify credentials in `.env`
   - Check endpoint URLs
   - Verify API keys
   - Ensure proper Azure permissions

3. **Index Not Found**: Create index first
   ```bash
   python scripts/create_index.py
   ```

4. **Conversation Memory Not Working**: 
   - Verify `AZURE_PROJECT_ENDPOINT` and `AZURE_AI_AGENT_ID` are set
   - Check Azure AI Projects package is installed
   - Ensure Azure credentials have proper permissions

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on the repository.

---

**Built with ❤️ using FastAPI, Azure AI Services, and LangChain**
