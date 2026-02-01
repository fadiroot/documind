"""Retrieval service for RAG operations."""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool

from core.services.retrieval.embedding_service import EmbeddingService
from core.services.retrieval.search_service import SearchService
from core.services.retrieval.retrieval_result import RetrievalResult
from core.utils.logger import logger


class RetrievalService:
    def __init__(self, min_score_threshold: float = 0.3, enable_reranking: bool = False):
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
        self.min_score_threshold = min_score_threshold
        self.enable_reranking = enable_reranking

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        filters: Optional[str] = None,
        max_context_length: int = 8000
    ) -> RetrievalResult:
        min_score = min_score or self.min_score_threshold
        query_embedding = self.embedding_service.create_embedding(query)
        if not query_embedding:
            logger.error("Failed to create query embedding")
            return RetrievalResult(
                documents=[], scores=[], metadata=[], total_found=0, filtered_count=0
            )

        initial_k = top_k * 3 if min_score > 0 else top_k
        results = self.search_service.search(
            query_vector=query_embedding, top_k=initial_k, filters=filters
        )

        documents = []
        scores = []
        metadata_list = []
        total_length = 0
        filtered_count = 0
        all_scores = []

        for result in results:
            score = result.get("@search.score", 0.0)
            all_scores.append(score)
            if score < min_score:
                filtered_count += 1
                continue
            content = result.get("content", "")
            if total_length + len(content) > max_context_length:
                filtered_count += 1
                continue
            doc_metadata = self._build_document_metadata(result, score)
            doc = Document(page_content=content, metadata=doc_metadata)
            documents.append(doc)
            scores.append(score)
            metadata_list.append(doc_metadata)
            total_length += len(content)
            if len(documents) >= top_k:
                break

        if len(documents) == 0 and len(results) > 0 and all_scores:
            relaxed_threshold = max(0.3, max(all_scores) * 0.5)
            for result in results:
                score = result.get("@search.score", 0.0)
                if score >= relaxed_threshold:
                    content = result.get("content", "")
                    if total_length + len(content) <= max_context_length:
                        doc_metadata = self._build_document_metadata(result, score)
                        doc = Document(page_content=content, metadata=doc_metadata)
                        documents.append(doc)
                        scores.append(score)
                        metadata_list.append(doc_metadata)
                        total_length += len(content)
                        if len(documents) >= top_k:
                            break

        if self.enable_reranking and len(documents) > 1:
            documents, scores = self._rerank_documents(query, documents, scores)

        return RetrievalResult(
            documents=documents,
            scores=scores,
            metadata=metadata_list,
            total_found=len(results),
            filtered_count=filtered_count
        )

    def _build_document_metadata(self, result: Dict[str, Any], score: float) -> Dict[str, Any]:
        source_doc = result.get("source_document") or result.get("document_name", "")
        return {
            "id": result.get("id"),
            "document_name": result.get("document_name") or source_doc,
            "source_document": source_doc,
            "document_title": result.get("document_title"),
            "page_number": result.get("page_number"),
            "chunk_index": result.get("chunk_index"),
            "token_count": result.get("token_count"),
            "score": score,
            "legal_part_name": result.get("legal_part_name"),
            "legal_chapter_name": result.get("legal_chapter_name"),
            "article_reference": result.get("article_reference"),
            "article_number": result.get("article_number"),
            "paragraph_number": result.get("paragraph_number"),
            "clause_number": result.get("clause_number"),
            "procedure_name": result.get("procedure_name"),
            "procedure_step": result.get("procedure_step"),
            "policy_name": result.get("policy_name"),
            "annex_name": result.get("annex_name"),
            "rank": result.get("rank"),
            "grade": result.get("grade"),
            "category_class": result.get("category_class"),
            "group": result.get("group"),
            "cadre_classification": result.get("cadre_classification"),
            "category": result.get("category"),
            "target_audience": result.get("target_audience"),
            "keywords": result.get("keywords"),
            "metadata_item_number": result.get("metadata_item_number"),
            "metadata_item_type": result.get("metadata_item_type"),
            "metadata_item_title": result.get("metadata_item_title"),
            "metadata_section_title": result.get("metadata_section_title"),
            "metadata_resource_path": result.get("metadata_resource_path"),
            "metadata_source_file": source_doc,
        }

    def _rerank_documents(
        self, query: str, documents: List[Document], scores: List[float]
    ) -> tuple[List[Document], List[float]]:
        return documents, scores


def create_retrieve_tool(retrieval_service: RetrievalService):
    @tool
    def retrieve_tool(
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        filters: Optional[str] = None,
        max_context_length: int = 8000,
    ) -> str:
        """Retrieve relevant documents. Use when the question needs document retrieval (regulations, articles, legal content). Leave filters empty unless you have a valid OData expression like field eq 'value'."""
        result = retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
            max_context_length=max_context_length,
        )
        parts = []
        for i, (doc, score, meta) in enumerate(zip(result.documents, result.scores, result.metadata)):
            doc_name = meta.get("document_name", meta.get("id", "?"))
            parts.append(f"[{i + 1}] {doc_name} (score: {score:.2f}):\n{doc.page_content}")
        return "\n\n".join(parts) if parts else "No relevant documents found."
    return retrieve_tool
