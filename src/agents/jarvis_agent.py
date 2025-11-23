"""Main Jarvis assistant agent - orchestrates all other agents."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from src.agents.base_agent import BaseAgent, AgentStatus
from src.core.logger import setup_logger
from src.core.mcp_protocol import mcp_handler, MCPMessage, MessageType, MessagePriority

# Import specialized agents
from src.agents.researcher_agent import ResearcherAgent
from src.agents.scheduler_agent import SchedulerAgent
from src.agents.email_agent import EmailAgent
from src.agents.iot_controller_agent import IoTControllerAgent
from src.agents.data_analyst_agent import DataAnalystAgent

logger = setup_logger(__name__)


class JarvisAgent(BaseAgent):
    """Main Jarvis assistant - coordinates specialized agents and handles general queries."""

    def __init__(self, memory=None, llm=None):
        """Initialize Jarvis agent."""
        super().__init__(
            name="Jarvis",
            description="AI personal assistant with multi-modal capabilities and task orchestration",
        )
        self.specialized_agents = {}
        self.conversation_context = []
        self.mcp_handler = mcp_handler

        # Initialize specialized agents
        self._initialize_specialized_agents(memory, llm)

    def _initialize_specialized_agents(self, memory=None, llm=None):
        """Initialize all specialized agents."""
        try:
            # Create specialized agents
            agents = {
                "researcher": ResearcherAgent(memory=memory, llm=llm),
                "scheduler": SchedulerAgent(memory=memory, llm=llm),
                "email": EmailAgent(memory=memory, llm=llm),
                "iot_controller": IoTControllerAgent(memory=memory, llm=llm),
                "analyst": DataAnalystAgent(memory=memory, llm=llm),
            }

            # Register each agent
            for agent_type, agent in agents.items():
                self.register_agent(agent_type, agent)

                # Register with MCP handler
                capabilities = [
                    {
                        "name": cap,
                        "description": f"{agent.name} capability: {cap}",
                        "input_schema": {},
                        "output_schema": {}
                    }
                    for cap in agent.capabilities
                ]

                self.mcp_handler.register_agent(
                    agent_id=agent_type,
                    name=agent.name,
                    agent_type=agent_type,
                    capabilities=capabilities
                )

            logger.info(f"Initialized {len(agents)} specialized agents")

        except Exception as e:
            logger.error(f"Failed to initialize specialized agents: {e}")

    def register_agent(self, agent_type: str, agent: BaseAgent):
        """
        Register a specialized agent.

        Args:
            agent_type: Type identifier for the agent
            agent: Agent instance
        """
        self.specialized_agents[agent_type] = agent
        self.logger.info(f"Registered agent: {agent_type} ({agent.name})")

    def get_capabilities(self) -> List[str]:
        """Get Jarvis capabilities."""
        base_capabilities = [
            "Natural language conversation",
            "Context retention and memory",
            "Task decomposition and planning",
            "Multi-agent orchestration",
            "Proactive assistance",
        ]

        # Add capabilities from specialized agents
        for agent_type, agent in self.specialized_agents.items():
            base_capabilities.extend(
                [f"{agent_type}: {cap}" for cap in agent.get_capabilities()]
            )

        return base_capabilities

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task, delegating to specialized agents if needed.

        Args:
            task: Task dictionary with 'query', 'context', etc.

        Returns:
            Execution result
        """
        self.status = AgentStatus.EXECUTING
        start_time = datetime.utcnow()

        try:
            query = task.get("query", "")
            context = task.get("context", {})

            self.logger.info(f"Executing task: {query[:100]}...")

            # Add to conversation context
            self.conversation_context.append(
                {
                    "role": "user",
                    "content": query,
                    "timestamp": start_time.isoformat(),
                }
            )

            # Determine if we need specialized agents
            agent_routing = await self._route_task(query, context)

            if agent_routing["requires_specialized_agent"]:
                if agent_routing.get("multi_agent", False):
                    # Orchestrate multiple agents
                    result = await self._orchestrate_multi_agent_task(query, context)
                else:
                    # Delegate to specialized agent
                    result = await self._delegate_to_agent(
                        agent_routing["agent_type"], task
                    )
            else:
                # Handle with general capabilities
                result = await self._handle_general_query(query, context)

            # Store in memory
            self.memory.add_to_short_term(
                content=query,
                metadata={
                    "type": "user_query",
                    "timestamp": start_time.isoformat(),
                },
            )

            self.memory.add_to_short_term(
                content=result["response"],
                metadata={
                    "type": "assistant_response",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Add to conversation context
            self.conversation_context.append(
                {
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Limit context size
            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]

            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds()

            response = {
                "success": True,
                "response": result["response"],
                "metadata": {
                    "execution_time": execution_time,
                    "agent_used": result.get("agent_used", "jarvis"),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            self.log_execution(task, response)
            self.status = AgentStatus.COMPLETED
            return response

        except Exception as e:
            self.logger.error(f"Error executing task: {e}", exc_info=True)
            self.status = AgentStatus.FAILED

            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _route_task(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine which agent should handle the task.

        Args:
            query: User query
            context: Additional context

        Returns:
            Routing decision
        """
        # Enhanced routing with multi-agent support
        query_lower = query.lower()

        # Check for calendar/scheduling
        if any(
            kw in query_lower
            for kw in ["schedule", "calendar", "meeting", "appointment", "remind"]
        ):
            return {
                "requires_specialized_agent": True,
                "agent_type": "scheduler",
                "multi_agent": False
            }

        # Check for email
        if any(kw in query_lower for kw in ["email", "send", "inbox", "draft", "reply"]):
            return {
                "requires_specialized_agent": True,
                "agent_type": "email",
                "multi_agent": False
            }

        # Check for web search/research
        if any(
            kw in query_lower for kw in ["search", "find", "look up", "what is", "research"]
        ):
            return {
                "requires_specialized_agent": True,
                "agent_type": "researcher",
                "multi_agent": False
            }

        # Check for data analysis
        if any(kw in query_lower for kw in ["analyze", "data", "chart", "graph", "statistics"]):
            return {
                "requires_specialized_agent": True,
                "agent_type": "analyst",
                "multi_agent": False
            }

        # Check for IoT/device control
        if any(
            kw in query_lower
            for kw in ["light", "temperature", "device", "lock", "unlock", "camera", "home"]
        ):
            return {
                "requires_specialized_agent": True,
                "agent_type": "iot_controller",
                "multi_agent": False
            }

        # Check for complex multi-agent tasks
        if any(
            kw in query_lower
            for kw in ["plan", "coordinate", "organize", "comprehensive"]
        ):
            return {
                "requires_specialized_agent": True,
                "agent_type": "multi_agent",
                "multi_agent": True
            }

        # Default to general handling
        return {
            "requires_specialized_agent": False,
            "agent_type": "jarvis",
            "multi_agent": False
        }

    async def _delegate_to_agent(
        self, agent_type: str, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delegate task to specialized agent.

        Args:
            agent_type: Type of agent to use
            task: Task to execute

        Returns:
            Agent result
        """
        if agent_type not in self.specialized_agents:
            self.logger.warning(
                f"Agent {agent_type} not registered, falling back to general"
            )
            return await self._handle_general_query(
                task.get("query", ""), task.get("context", {})
            )

        agent = self.specialized_agents[agent_type]
        self.logger.info(f"Delegating to {agent.name}")

        result = await agent.execute(task)
        return {
            "response": result.get("response", ""),
            "agent_used": agent.name,
        }

    async def _orchestrate_multi_agent_task(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrate complex tasks requiring multiple agents.

        Args:
            query: User query
            context: Additional context

        Returns:
            Aggregated response from multiple agents
        """
        logger.info(f"Orchestrating multi-agent task: {query}")

        # Use LLM to break down the task
        decomposition_prompt = f"""
        Break down this complex task into subtasks for specialized agents:
        Task: {query}

        Available agents:
        - researcher: Web search, fact-checking, research
        - scheduler: Calendar, meetings, reminders
        - email: Email management, drafting
        - iot_controller: Device control, home automation
        - analyst: Data analysis, visualization

        Return JSON with subtasks and required agents.
        Example: {{"subtasks": [{{"agent": "researcher", "task": "..."}}, ...]}}
        """

        # Generate task decomposition
        task_plan = await self.think(decomposition_prompt)

        # Execute subtasks (simplified - in production would parse JSON properly)
        results = []

        # For demo, route to relevant agents based on keywords
        if any(kw in query.lower() for kw in ["research", "find", "search"]):
            researcher = self.specialized_agents.get("researcher")
            if researcher:
                res = await researcher.process({
                    "action": "research",
                    "query": query
                })
                results.append(("Research", res))

        if any(kw in query.lower() for kw in ["schedule", "calendar"]):
            scheduler = self.specialized_agents.get("scheduler")
            if scheduler:
                res = await scheduler.process({
                    "action": "list_events",
                    "days_ahead": 7
                })
                results.append(("Schedule", res))

        if any(kw in query.lower() for kw in ["analyze", "data"]):
            analyst = self.specialized_agents.get("analyst")
            if analyst:
                res = await analyst.process({
                    "action": "analyze",
                    "data": context.get("data", {})
                })
                results.append(("Analysis", res))

        # Synthesize results
        synthesis_prompt = f"""
        Synthesize these agent results into a cohesive response:
        Original query: {query}

        Results:
        {results}

        Provide a unified, helpful response to the user.
        """

        final_response = await self.think(synthesis_prompt)

        return {
            "response": final_response,
            "agent_used": "jarvis (multi-agent)",
            "agents_involved": [r[0] for r in results]
        }

    async def _handle_general_query(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle general query with LLM.

        Args:
            query: User query
            context: Additional context

        Returns:
            Response
        """
        # Build context from conversation history
        conversation_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_context[-10:]
            ]
        )

        # Query memory for relevant information
        relevant_memories = self.memory.query_long_term(query, n_results=3)
        memory_context = "\n".join(
            [mem["documents"][0] for mem in relevant_memories.get("documents", [[]])]
        )

        # Build full context
        full_context = ""
        if conversation_history:
            full_context += f"Recent conversation:\n{conversation_history}\n\n"
        if memory_context:
            full_context += f"Relevant information:\n{memory_context}\n\n"

        # Generate response
        response = await self.think(query, full_context if full_context else None)

        return {
            "response": response,
            "agent_used": "jarvis",
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_context.copy()

    def clear_conversation(self):
        """Clear conversation context."""
        self.conversation_context.clear()
        self.logger.info("Conversation context cleared")
