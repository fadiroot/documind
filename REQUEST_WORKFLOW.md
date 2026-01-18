# Request Workflow - DocuMind RAG System

## 📋 Complete Request Flow: From API Call to Response

### **1. API Request Entry Point**

**File:** `app/main.py`
- FastAPI application receives HTTP request
- Routes to appropriate endpoint based on URL path

---

## 🔄 **Q&A Request Flow** (`POST /api/qa/ask`)

### **Step 1: Request Reception**
```
HTTP POST /api/qa/ask
↓
app/routes/qa.py → ask_question()
```

**Function:** `ask_question()` (lines 15-61)
- Receives `QuestionRequest` with:
  - `question`: User's question text
  - `context_ids`: Optional document IDs to filter
  - `category`: Optional category (e.g., "legal", "financial")
- Query parameters:
  - `session_id`: For conversation memory
  - `use_conversation`: Enable conversation history

---

### **Step 2: Agent Service Processing**
```
agent_service.answer_question()
↓
core/services/agent_service.py → answer_question() (lines 216-288)
```

**Function Flow:**
1. **Input Preparation** (line 247-249)
   ```python
   chain_input = {"input": question}
   if category:
       chain_input["category"] = category
   ```

2. **Chain Selection** (lines 252-268)
   - **If `use_conversation=True` and `session_id` provided:**
     - Calls `_create_conversational_chain()` (line 253)
     - Creates chain with conversation memory
   - **Else:**
     - Uses regular `qa_chain` (line 267)

---

### **Step 3: Retrieval Process**

#### **3a. Query Embedding**
```
agent_service.qa_chain.invoke(chain_input)
↓
RunnableLambda(get_query) → extracts question
↓
self.retriever → AzureAISearchRetriever._get_relevant_documents()
↓
core/services/retrieval_service.py → retrieve()
```

**Function:** `RetrievalService.retrieve()` (lines 15-52)

**Calls:**
1. **`embedding_service.create_embedding(query)`** (line 28)
   - **File:** `core/services/embedding_service.py`
   - Converts query text to embedding vector
   - Uses Azure OpenAI Embeddings API

2. **`vectorstore_service.search(query_vector, top_k)`** (line 34)
   - **File:** `core/services/vectorstore_service.py`
   - **Function:** `VectorStoreService.search()` (lines 38-79)
   - Creates `VectorizedQuery` with embedding
   - Searches Azure AI Search index
   - Returns top_k similar documents

---

#### **3b. Document Formatting**
```
RetrievalService.retrieve() → format_docs()
↓
Converts List[Document] to formatted string
```

**Function:** `format_docs()` (lines 100-101, 158-159)
- Joins document contents with `"\n\n"`
- Returns formatted context string

---

### **Step 4: Prompt Template Application**

**Function:** `ChatPromptTemplate.from_template()` or `ChatPromptTemplate.from_messages()`

**Templates Used:**

1. **Basic QA Template** (lines 87-97):
   ```python
   qa_template = """Use the following pieces of context...
   Context: {context}
   Question: {input}
   Helpful Answer:"""
   ```

2. **Conversational Template** (lines 183-187):
   ```python
   qa_prompt = ChatPromptTemplate.from_messages([
       ("system", qa_system_prompt),
       MessagesPlaceholder(variable_name="chat_history"),
       ("human", "{input}"),
   ])
   ```

**Variables Filled:**
- `{context}`: Formatted documents from retrieval
- `{input}`: User's question
- `{category_instruction}`: Category-specific instructions (if provided)
- `{chat_history}`: Previous conversation messages (if conversational)

---

### **Step 5: LLM Generation**

```
qa_prompt | self.llm | StrOutputParser()
↓
AzureChatOpenAI (Azure OpenAI GPT-4o)
↓
Generates answer based on context + question
```

**Function:** `AzureChatOpenAI.invoke()` or `.stream()`
- **File:** `core/services/agent_service.py` (lines 61-68)
- Model: `gpt-4o` (configurable)
- Temperature: `0.3`
- Streaming: `True`

---

### **Step 6: Response Processing**

**Function:** `answer_question()` continues (lines 256-281)

1. **Get Source Documents** (line 257, 264, 268)
   ```python
   source_docs = self.retriever._get_relevant_documents(question)
   ```

2. **Format Sources** (line 271)
   ```python
   sources = self._format_source_documents(source_docs)
   ```
   - **Function:** `_format_source_documents()` (lines 362-374)
   - Extracts: id, content preview, document_name, page_number, chunk_index, score

3. **Calculate Confidence** (line 274)
   ```python
   confidence = self._calculate_confidence_from_docs(source_docs)
   ```
   - **Function:** `_calculate_confidence_from_docs()` (lines 376-391)
   - Averages top 3 document scores
   - Normalizes to 0-1 range

4. **Update Memory** (if conversational) (lines 259-260)
   ```python
   memory.append({"role": "user", "content": question})
   memory.append({"role": "assistant", "content": answer})
   ```

---

### **Step 7: Response Formatting**

**Function:** `ask_question()` in `app/routes/qa.py` (lines 41-55)

1. **Create Response Model** (lines 41-45)
   ```python
   response = QuestionResponse(
       answer=result.get("answer", ""),
       sources=result.get("sources", []),
       confidence=result.get("confidence")
   )
   ```

2. **Wrap in API Response** (lines 51-55)
   ```python
   return APIResponse(
       success=True,
       message="Question answered successfully",
       data=response_data
   )
   ```

---

### **Step 8: HTTP Response**

```
FastAPI returns JSON response:
{
    "success": true,
    "message": "Question answered successfully",
    "data": {
        "answer": "...",
        "sources": [...],
        "confidence": 0.85,
        "session_id": "..." (if provided)
    }
}
```

---

## 🔄 **Streaming Request Flow** (`POST /api/qa/ask/stream`)

### **Similar Flow with Differences:**

1. **Entry:** `ask_question_stream()` (lines 64-106)
2. **Calls:** `agent_service.stream_answer()` (lines 290-360)
3. **Uses:** `chain.stream()` instead of `chain.invoke()`
4. **Returns:** `StreamingResponse` with Server-Sent Events (SSE)
5. **Format:** `data: {"chunk": "..."}\n\n`

---

## 📊 **Complete Function Call Chain**

```
HTTP Request
    ↓
app/main.py (FastAPI)
    ↓
app/routes/qa.py::ask_question()
    ↓
core/services/agent_service.py::answer_question()
    ↓
    ├─→ _create_conversational_chain() [if use_conversation]
    │   └─→ _create_simple_conversational_chain()
    │       └─→ _get_or_create_memory()
    │       └─→ _convert_history_to_messages()
    │
    └─→ qa_chain.invoke()
        ↓
        ├─→ RunnableLambda(get_query)
        │   └─→ Extracts question from input
        │
        ├─→ self.retriever (AzureAISearchRetriever)
        │   └─→ _get_relevant_documents()
        │       └─→ RetrievalService.retrieve()
        │           ├─→ EmbeddingService.create_embedding()
        │           │   └─→ Azure OpenAI Embeddings API
        │           │
        │           └─→ VectorStoreService.search()
        │               └─→ Azure AI Search API
        │
        ├─→ RunnableLambda(format_docs)
        │   └─→ Formats documents to string
        │
        ├─→ ChatPromptTemplate
        │   └─→ Fills template with context + question
        │
        └─→ AzureChatOpenAI.invoke()
            └─→ Azure OpenAI GPT-4o API
                └─→ Returns answer
    ↓
_format_source_documents()
_calculate_confidence_from_docs()
    ↓
app/routes/qa.py::ask_question()
    ↓
APIResponse
    ↓
HTTP JSON Response
```

---

## 🔑 **Key Components**

### **Services:**
1. **AgentService** - Main orchestration, RAG chains
2. **RetrievalService** - Document retrieval coordination
3. **EmbeddingService** - Text to vector conversion
4. **VectorStoreService** - Azure AI Search operations

### **LangChain Components:**
1. **AzureAISearchRetriever** - Custom retriever wrapper
2. **ChatPromptTemplate** - Prompt management
3. **AzureChatOpenAI** - LLM interface
4. **RunnableLambda** - Custom processing functions
5. **StrOutputParser** - Response parsing

### **Data Flow:**
```
Question (text)
    ↓
Embedding (vector)
    ↓
Vector Search (Azure AI Search)
    ↓
Retrieved Documents (context)
    ↓
Prompt Template (context + question)
    ↓
LLM (Azure OpenAI)
    ↓
Answer (text) + Sources + Confidence
```

---

## 📝 **Example Request/Response**

### **Request:**
```json
POST /api/qa/ask?session_id=user123&use_conversation=true
{
    "question": "What are the vacation policies?",
    "category": "legal"
}
```

### **Response:**
```json
{
    "success": true,
    "message": "Question answered successfully",
    "data": {
        "answer": "According to the documents...",
        "sources": [
            {
                "id": "doc1_0",
                "document_name": "الإجازات.pdf",
                "page_number": 5,
                "score": 0.92
            }
        ],
        "confidence": 0.92,
        "session_id": "user123"
    }
}
```
