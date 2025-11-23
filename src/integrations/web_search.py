"""Web search integration using Serper and Tavily APIs."""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class WebSearchTool:
    """Web search tool supporting multiple search providers."""

    def __init__(self):
        """Initialize web search tool."""
        self.serper_key = settings.serper_api_key
        self.tavily_key = settings.tavily_api_key
        self.logger = setup_logger(__name__)

    async def search(
        self,
        query: str,
        num_results: int = 5,
        provider: str = "serper",
    ) -> List[Dict[str, Any]]:
        """
        Perform web search.

        Args:
            query: Search query
            num_results: Number of results to return
            provider: Search provider ('serper' or 'tavily')

        Returns:
            List of search results
        """
        if provider == "serper" and self.serper_key:
            return await self._search_serper(query, num_results)
        elif provider == "tavily" and self.tavily_key:
            return await self._search_tavily(query, num_results)
        else:
            # Fallback to simple search
            return await self._search_fallback(query, num_results)

    async def _search_serper(
        self, query: str, num_results: int
    ) -> List[Dict[str, Any]]:
        """Search using Serper API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={
                        "q": query,
                        "num": num_results,
                    },
                    headers={
                        "X-API-KEY": self.serper_key,
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic", [])[:num_results]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "serper",
                        }
                    )

                self.logger.info(f"Serper search returned {len(results)} results")
                return results

        except Exception as e:
            self.logger.error(f"Serper search failed: {e}")
            return []

    async def _search_tavily(
        self, query: str, num_results: int
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_key,
                        "query": query,
                        "max_results": num_results,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("results", [])[:num_results]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", ""),
                            "source": "tavily",
                        }
                    )

                self.logger.info(f"Tavily search returned {len(results)} results")
                return results

        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            return []

    async def _search_fallback(
        self, query: str, num_results: int
    ) -> List[Dict[str, Any]]:
        """Fallback search (mock for now)."""
        self.logger.warning("Using fallback search - no API keys configured")
        return [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com",
                "snippet": "Web search API not configured. Please add SERPER_API_KEY or TAVILY_API_KEY to .env",
                "source": "fallback",
            }
        ]

    async def summarize_results(
        self, results: List[Dict[str, Any]], llm_client=None
    ) -> str:
        """
        Summarize search results using LLM.

        Args:
            results: Search results to summarize
            llm_client: LLM client for summarization

        Returns:
            Summary text
        """
        if not results:
            return "No search results found."

        # Create context from results
        context = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['title']}\n"
            context += f"   {result['snippet']}\n"
            context += f"   URL: {result['url']}\n\n"

        if llm_client:
            from src.core.llm import get_llm_client

            llm = llm_client or get_llm_client()
            summary = await llm.generate(
                prompt=f"Summarize these search results concisely:\n\n{context}",
                system_prompt="You are a helpful assistant that summarizes search results.",
            )
            return summary
        else:
            return context


class WebScraperTool:
    """Tool for scraping web page content."""

    async def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from URL.

        Args:
            url: URL to scrape

        Returns:
            Scraped content
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()

                # Simple text extraction (can be enhanced with BeautifulSoup)
                content = response.text

                return {
                    "url": url,
                    "content": content[:5000],  # Limit content size
                    "status_code": response.status_code,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
