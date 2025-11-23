"""Memory management system using ChromaDB for vector storage."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class MemoryManager:
    """Manages short-term and long-term memory using vector database."""

    def __init__(self):
        """Initialize ChromaDB client and collections."""
        self.client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        # Create collections
        self.short_term = self.client.get_or_create_collection(
            name="short_term_memory",
            metadata={"description": "Recent conversation context"},
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"description": "User preferences and historical data"},
        )

        self.knowledge = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Learned facts and information"},
        )

        logger.info("Memory manager initialized with ChromaDB")

    def add_to_short_term(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add content to short-term memory.

        Args:
            content: Text content to store
            metadata: Optional metadata dictionary
            embedding: Optional pre-computed embedding

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        meta["timestamp"] = datetime.utcnow().isoformat()
        meta["type"] = "conversation"

        if embedding:
            self.short_term.add(
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
                embeddings=[embedding],
            )
        else:
            self.short_term.add(
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
            )

        logger.debug(f"Added to short-term memory: {doc_id}")
        return doc_id

    def add_to_long_term(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add content to long-term memory.

        Args:
            content: Text content to store
            metadata: Optional metadata dictionary

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        meta["timestamp"] = datetime.utcnow().isoformat()

        self.long_term.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id],
        )

        logger.debug(f"Added to long-term memory: {doc_id}")
        return doc_id

    def query_short_term(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query short-term memory.

        Args:
            query: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """
        results = self.short_term.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results

    def query_long_term(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query long-term memory.

        Args:
            query: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """
        results = self.long_term.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results

    def get_recent_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.

        Args:
            limit: Maximum number of recent items

        Returns:
            List of recent memories
        """
        try:
            results = self.short_term.get(limit=limit)
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return []

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.client.delete_collection("short_term_memory")
        self.short_term = self.client.create_collection("short_term_memory")
        logger.info("Short-term memory cleared")

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB results."""
        formatted = []
        if not results or "documents" not in results:
            return formatted

        for i, doc in enumerate(results["documents"]):
            formatted.append(
                {
                    "content": doc,
                    "metadata": results.get("metadatas", [])[i]
                    if i < len(results.get("metadatas", []))
                    else {},
                    "distance": results.get("distances", [])[i]
                    if i < len(results.get("distances", []))
                    else None,
                }
            )

        return formatted

    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "short_term_count": self.short_term.count(),
            "long_term_count": self.long_term.count(),
            "knowledge_count": self.knowledge.count(),
        }


# Global memory instance
memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager()
    return memory_manager
