"""RAG (Retrieval-Augmented Generation) tool implementation."""

import logging
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from .base_tool import BaseTool, ToolConfig, ToolResult


class RAGTool(BaseTool):
    """RAG tool for document retrieval and augmented generation."""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.collection_name = config.parameters.get("collection_name", "default_collection")
        self.embedding_model_name = config.parameters.get("embedding_model", "all-MiniLM-L6-v2")
        self.persist_directory = config.parameters.get("persist_directory", "./chroma_db")
        self.top_k = config.parameters.get("top_k", 5)
        self.similarity_threshold = config.parameters.get("similarity_threshold", 0.7)
    
    async def initialize(self) -> None:
        """Initialize the RAG tool with ChromaDB and embedding model."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                self.collection = self.client.create_collection(name=self.collection_name)
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG tool: {str(e)}")
            raise
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute RAG operations (add documents, query, or retrieve)."""
        await self._ensure_initialized()
        
        operation = kwargs.get("operation", "query")
        
        try:
            if operation == "add_documents":
                return await self._add_documents(kwargs)
            elif operation == "query":
                return await self._query_documents(kwargs)
            elif operation == "retrieve":
                return await self._retrieve_documents(kwargs)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        
        except Exception as e:
            self.logger.error(f"RAG tool execution failed: {str(e)}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def _add_documents(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add documents to the RAG collection."""
        documents = kwargs.get("documents", [])
        metadatas = kwargs.get("metadatas", [])
        ids = kwargs.get("ids", [])
        
        if not documents:
            return ToolResult(
                success=False,
                error="No documents provided"
            )
        
        # Generate IDs if not provided
        if not ids:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas if metadatas else None,
            ids=ids
        )
        
        return ToolResult(
            success=True,
            data={"added_documents": len(documents)},
            metadata={"collection_name": self.collection_name}
        )
    
    async def _query_documents(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Query documents using similarity search."""
        query = kwargs.get("query", "")
        n_results = kwargs.get("n_results", self.top_k)
        
        if not query:
            return ToolResult(
                success=False,
                error="No query provided"
            )
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Process results
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        
        # Filter by similarity threshold
        filtered_results = []
        for i, distance in enumerate(distances):
            similarity = 1 - distance  # Convert distance to similarity
            if similarity >= self.similarity_threshold:
                filtered_results.append({
                    "document": documents[i],
                    "similarity": similarity,
                    "metadata": metadatas[i] if metadatas else {},
                    "id": ids[i]
                })
        
        return ToolResult(
            success=True,
            data={
                "results": filtered_results,
                "total_found": len(filtered_results)
            },
            metadata={
                "query": query,
                "similarity_threshold": self.similarity_threshold
            }
        )
    
    async def _retrieve_documents(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Retrieve documents by IDs."""
        ids = kwargs.get("ids", [])
        
        if not ids:
            return ToolResult(
                success=False,
                error="No document IDs provided"
            )
        
        # Get documents by IDs
        results = self.collection.get(ids=ids)
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        retrieved_docs = []
        for i, doc in enumerate(documents):
            retrieved_docs.append({
                "document": doc,
                "metadata": metadatas[i] if metadatas else {},
                "id": ids[i]
            })
        
        return ToolResult(
            success=True,
            data={
                "documents": retrieved_docs,
                "total_retrieved": len(retrieved_docs)
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add_documents", "query", "retrieve"],
                    "description": "Operation to perform"
                },
                "query": {
                    "type": "string",
                    "description": "Query string for similarity search"
                },
                "documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Documents to add to the collection"
                },
                "metadatas": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Metadata for documents"
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["operation"]
        }
