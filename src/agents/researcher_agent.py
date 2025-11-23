"""Research agent for web search and information gathering."""

from typing import Dict, Any, List
from datetime import datetime

from src.agents.base_agent import BaseAgent, AgentStatus
from src.integrations.web_search import WebSearchTool, WebScraperTool
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class ResearcherAgent(BaseAgent):
    """Specialized agent for web research and information gathering."""

    def __init__(self):
        """Initialize researcher agent."""
        super().__init__(
            name="Researcher",
            description="Performs web searches, gathers information, and synthesizes research",
        )
        self.search_tool = WebSearchTool()
        self.scraper_tool = WebScraperTool()

    def get_capabilities(self) -> List[str]:
        """Get researcher capabilities."""
        return [
            "Web search across multiple providers",
            "Information synthesis and summarization",
            "URL content scraping",
            "Fact checking and verification",
            "Research compilation",
        ]

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task.

        Args:
            task: Task with 'query', 'num_results', 'include_summary'

        Returns:
            Research results
        """
        self.status = AgentStatus.EXECUTING
        start_time = datetime.utcnow()

        try:
            query = task.get("query", "")
            num_results = task.get("num_results", 5)
            include_summary = task.get("include_summary", True)
            scrape_urls = task.get("scrape_urls", False)

            self.logger.info(f"Researching: {query}")

            # Perform web search
            search_results = await self.search_tool.search(
                query=query,
                num_results=num_results,
            )

            # Optionally scrape URLs
            scraped_content = []
            if scrape_urls and search_results:
                for result in search_results[:3]:  # Scrape top 3 only
                    content = await self.scraper_tool.scrape(result["url"])
                    if "error" not in content:
                        scraped_content.append(content)

            # Generate summary
            summary = ""
            if include_summary:
                summary = await self.search_tool.summarize_results(
                    search_results, self.llm
                )

            # Store in memory
            memory_content = f"Research query: {query}\nResults: {len(search_results)} sources found"
            if summary:
                memory_content += f"\nSummary: {summary[:200]}"

            self.remember(
                memory_content,
                metadata={
                    "type": "research",
                    "query": query,
                    "num_results": len(search_results),
                },
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            result = {
                "success": True,
                "query": query,
                "search_results": search_results,
                "summary": summary,
                "scraped_content": scraped_content,
                "num_results": len(search_results),
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Build response text
            response_text = f"Found {len(search_results)} results for '{query}'.\n\n"
            if summary:
                response_text += f"Summary:\n{summary}\n\n"

            response_text += "Top results:\n"
            for i, res in enumerate(search_results[:3], 1):
                response_text += f"{i}. {res['title']}\n   {res['url']}\n   {res['snippet'][:100]}...\n\n"

            result["response"] = response_text

            self.log_execution(task, result)
            self.status = AgentStatus.COMPLETED

            return result

        except Exception as e:
            self.logger.error(f"Research failed: {e}", exc_info=True)
            self.status = AgentStatus.FAILED

            return {
                "success": False,
                "error": str(e),
                "response": f"Research failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def fact_check(self, claim: str) -> Dict[str, Any]:
        """
        Fact check a claim using web search.

        Args:
            claim: Claim to verify

        Returns:
            Fact check result
        """
        self.logger.info(f"Fact checking: {claim}")

        # Search for the claim
        search_results = await self.search_tool.search(
            query=f"fact check: {claim}",
            num_results=5,
        )

        # Analyze with LLM
        context = "\n".join(
            [
                f"- {res['title']}: {res['snippet']}"
                for res in search_results
            ]
        )

        analysis = await self.think(
            prompt=f"Based on these search results, evaluate if this claim is true: '{claim}'\n\nSources:\n{context}",
            context="You are a fact-checker. Provide a clear verdict (TRUE/FALSE/UNCLEAR) and explanation.",
        )

        return {
            "claim": claim,
            "verdict": analysis,
            "sources": search_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def research_topic(
        self, topic: str, depth: int = 3
    ) -> Dict[str, Any]:
        """
        Deep research on a topic with multiple queries.

        Args:
            topic: Topic to research
            depth: Number of related queries to explore

        Returns:
            Comprehensive research results
        """
        self.logger.info(f"Deep research on: {topic}")

        all_results = []

        # Main topic search
        main_results = await self.search_tool.search(topic, num_results=5)
        all_results.extend(main_results)

        # Generate related queries using LLM
        related_queries_text = await self.think(
            prompt=f"Generate {depth} related search queries for researching: {topic}",
            context="Return only the queries, one per line, without numbering.",
        )

        related_queries = [
            q.strip() for q in related_queries_text.split("\n") if q.strip()
        ][:depth]

        # Search related topics
        for query in related_queries:
            results = await self.search_tool.search(query, num_results=3)
            all_results.extend(results)

        # Synthesize findings
        synthesis = await self.search_tool.summarize_results(all_results, self.llm)

        return {
            "topic": topic,
            "main_results": main_results,
            "related_queries": related_queries,
            "all_results": all_results,
            "synthesis": synthesis,
            "timestamp": datetime.utcnow().isoformat(),
        }
