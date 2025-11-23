"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from src.core.llm import get_llm_client, LLMClient
from src.core.memory import get_memory_manager, MemoryManager
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_client: Optional[LLMClient] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            description: Agent description/purpose
            llm_client: Optional LLM client instance
            memory_manager: Optional memory manager instance
        """
        self.name = name
        self.description = description
        self.llm = llm_client or get_llm_client()
        self.memory = memory_manager or get_memory_manager()
        self.status = AgentStatus.IDLE
        self.logger = setup_logger(f"agent.{name}")
        self.execution_history: List[Dict[str, Any]] = []

        self.logger.info(f"Agent '{name}' initialized")

    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent's main task.

        Args:
            task: Task dictionary with parameters

        Returns:
            Execution result dictionary
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.

        Returns:
            List of capability descriptions
        """
        pass

    async def think(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Use LLM to reason about a problem.

        Args:
            prompt: Question or task
            context: Optional additional context

        Returns:
            LLM response
        """
        self.status = AgentStatus.THINKING
        self.logger.debug(f"Thinking about: {prompt[:100]}...")

        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nTask: {prompt}"

            response = await self.llm.generate(
                full_prompt,
                system_prompt=self._get_system_prompt(),
            )

            self.logger.debug(f"Thought complete: {response[:100]}...")
            return response

        except Exception as e:
            self.logger.error(f"Error in thinking: {e}")
            raise
        finally:
            self.status = AgentStatus.IDLE

    def _get_system_prompt(self) -> str:
        """
        Get agent-specific system prompt.

        Returns:
            System prompt string
        """
        return f"""You are {self.name}, a specialized AI agent.
Your purpose: {self.description}
Your capabilities: {', '.join(self.get_capabilities())}

Respond helpfully, accurately, and concisely."""

    def remember(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store information in memory.

        Args:
            content: Content to remember
            metadata: Optional metadata
        """
        meta = metadata or {}
        meta["agent"] = self.name
        meta["timestamp"] = datetime.utcnow().isoformat()

        self.memory.add_to_long_term(content, meta)
        self.logger.debug(f"Stored memory: {content[:50]}...")

    def recall(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.

        Args:
            query: Query text
            n_results: Number of results

        Returns:
            List of relevant memories
        """
        results = self.memory.query_long_term(query, n_results)
        self.logger.debug(f"Recalled {len(results)} memories for: {query[:50]}...")
        return results

    def log_execution(self, task: Dict[str, Any], result: Dict[str, Any]):
        """
        Log execution for debugging and learning.

        Args:
            task: Input task
            result: Execution result
        """
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "task": task,
            "result": result,
            "status": self.status.value,
        }
        self.execution_history.append(execution_record)

        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status.

        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "capabilities": self.get_capabilities(),
            "executions": len(self.execution_history),
        }

    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if task is appropriate for this agent.

        Args:
            task: Task dictionary

        Returns:
            True if valid, False otherwise
        """
        # Default implementation - can be overridden
        return True

    def reset(self):
        """Reset agent to initial state."""
        self.status = AgentStatus.IDLE
        self.execution_history.clear()
        self.logger.info(f"Agent '{self.name}' reset")
