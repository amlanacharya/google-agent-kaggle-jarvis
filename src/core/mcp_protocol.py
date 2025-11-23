"""
Model Context Protocol (MCP) Implementation.

Provides standardized communication between agents with:
- Message formatting and validation
- Context window management
- Token optimization
- Agent discovery and routing
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import uuid

from .logger import logger


class MessageType(str, Enum):
    """Message types in MCP protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MCPMessage(BaseModel):
    """Standard MCP message format."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    sender: str
    recipient: Optional[str] = None  # None for broadcast
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    payload: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None  # For threading
    requires_response: bool = False
    ttl: int = 300  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        return cls(**data)


class AgentCapability(BaseModel):
    """Agent capability descriptor."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    cost: float = 0.0  # Relative cost (0-1)
    latency: float = 0.0  # Expected latency in seconds


class AgentRegistration(BaseModel):
    """Agent registration information."""
    agent_id: str
    name: str
    type: str
    capabilities: List[AgentCapability]
    status: str = "active"
    load: float = 0.0  # Current load (0-1)
    priority: int = 0  # Routing priority


class MCPProtocolHandler:
    """
    MCP Protocol Handler for agent communication.

    Manages:
    - Agent registry
    - Message routing
    - Context management
    - Load balancing
    """

    def __init__(self):
        """Initialize MCP protocol handler."""
        self.agents: Dict[str, AgentRegistration] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.context_store: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[MCPMessage] = []
        self.max_history = 1000

        logger.info("MCP Protocol Handler initialized")

    def register_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        capabilities: List[Dict[str, Any]]
    ) -> bool:
        """Register an agent with the protocol handler."""
        try:
            cap_objects = [
                AgentCapability(**cap) for cap in capabilities
            ]

            registration = AgentRegistration(
                agent_id=agent_id,
                name=name,
                type=agent_type,
                capabilities=cap_objects
            )

            self.agents[agent_id] = registration
            logger.info(f"Registered agent: {name} ({agent_id})")

            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def discover_agents(
        self,
        capability: Optional[str] = None,
        agent_type: Optional[str] = None
    ) -> List[AgentRegistration]:
        """Discover agents by capability or type."""
        results = []

        for agent in self.agents.values():
            # Filter by type
            if agent_type and agent.type != agent_type:
                continue

            # Filter by capability
            if capability:
                has_capability = any(
                    cap.name == capability
                    for cap in agent.capabilities
                )
                if not has_capability:
                    continue

            results.append(agent)

        return results

    def find_best_agent(
        self,
        capability: str,
        prefer_low_load: bool = True
    ) -> Optional[str]:
        """Find best agent for a capability."""
        candidates = self.discover_agents(capability=capability)

        if not candidates:
            return None

        if prefer_low_load:
            # Sort by load (ascending)
            candidates.sort(key=lambda a: a.load)
        else:
            # Sort by priority (descending)
            candidates.sort(key=lambda a: a.priority, reverse=True)

        return candidates[0].agent_id

    async def send_message(
        self,
        message: MCPMessage,
        wait_for_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[MCPMessage]:
        """Send a message through MCP."""
        try:
            # Validate recipient exists
            if message.recipient and message.recipient not in self.agents:
                logger.error(f"Recipient not found: {message.recipient}")
                return None

            # Add to message history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history.pop(0)

            # Queue message for processing
            await self.message_queue.put(message)

            logger.debug(
                f"Message sent: {message.sender} -> {message.recipient}"
            )

            # Wait for response if required
            if wait_for_response:
                return await self._wait_for_response(message.id, timeout)

            return None

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    async def _wait_for_response(
        self,
        message_id: str,
        timeout: float
    ) -> Optional[MCPMessage]:
        """Wait for a response to a specific message."""
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"Timeout waiting for response to {message_id}")
                return None

            # Check message history for response
            for msg in reversed(self.message_history):
                if (msg.parent_id == message_id and
                    msg.type == MessageType.RESPONSE):
                    return msg

            await asyncio.sleep(0.1)

    async def broadcast_message(
        self,
        sender: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast a message to all agents."""
        message = MCPMessage(
            type=MessageType.BROADCAST,
            priority=MessagePriority.NORMAL,
            sender=sender,
            recipient=None,
            payload=payload,
            context=context
        )

        await self.send_message(message)

        return len(self.agents)

    def register_handler(
        self,
        agent_id: str,
        handler: Callable[[MCPMessage], Any]
    ):
        """Register a message handler for an agent."""
        self.message_handlers[agent_id] = handler
        logger.debug(f"Registered handler for {agent_id}")

    async def process_messages(self):
        """Process messages from the queue."""
        while True:
            try:
                message = await self.message_queue.get()

                # Handle broadcast
                if message.type == MessageType.BROADCAST:
                    for agent_id, handler in self.message_handlers.items():
                        if agent_id != message.sender:
                            await handler(message)

                # Handle directed message
                elif message.recipient in self.message_handlers:
                    handler = self.message_handlers[message.recipient]
                    await handler(message)
                else:
                    logger.warning(
                        f"No handler for recipient: {message.recipient}"
                    )

                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def store_context(
        self,
        context_id: str,
        context_data: Dict[str, Any]
    ):
        """Store context for later retrieval."""
        self.context_store[context_id] = {
            'data': context_data,
            'timestamp': datetime.now().timestamp()
        }

    def get_context(
        self,
        context_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve stored context."""
        if context_id in self.context_store:
            return self.context_store[context_id]['data']
        return None

    def clear_context(self, context_id: str):
        """Clear stored context."""
        if context_id in self.context_store:
            del self.context_store[context_id]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        return {
            'total_agents': len(self.agents),
            'agents': [
                {
                    'id': agent.agent_id,
                    'name': agent.name,
                    'type': agent.type,
                    'status': agent.status,
                    'load': agent.load,
                    'capabilities_count': len(agent.capabilities)
                }
                for agent in self.agents.values()
            ],
            'message_queue_size': self.message_queue.qsize(),
            'message_history_size': len(self.message_history)
        }

    def optimize_context_window(
        self,
        messages: List[MCPMessage],
        max_tokens: int = 4000
    ) -> List[MCPMessage]:
        """
        Optimize message context to fit within token limit.

        Strategy:
        1. Keep most recent messages
        2. Preserve high-priority messages
        3. Summarize older context
        """
        # Simple implementation: keep most recent
        # In production, would use tokenizer and smarter pruning

        optimized = []
        current_tokens = 0
        avg_message_tokens = 100  # Rough estimate

        # Sort by priority and recency
        sorted_messages = sorted(
            messages,
            key=lambda m: (m.priority.value, m.timestamp),
            reverse=True
        )

        for msg in sorted_messages:
            estimated_tokens = avg_message_tokens

            if current_tokens + estimated_tokens <= max_tokens:
                optimized.append(msg)
                current_tokens += estimated_tokens
            else:
                break

        # Re-sort by timestamp
        optimized.sort(key=lambda m: m.timestamp)

        return optimized

    async def request_capability_execution(
        self,
        capability: str,
        params: Dict[str, Any],
        sender: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Request execution of a capability from appropriate agent."""
        # Find best agent for capability
        agent_id = self.find_best_agent(capability)

        if not agent_id:
            logger.error(f"No agent found for capability: {capability}")
            return {
                'success': False,
                'error': f'No agent available for {capability}'
            }

        # Create request message
        message = MCPMessage(
            type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender=sender,
            recipient=agent_id,
            payload={
                'capability': capability,
                'params': params
            },
            context=context,
            requires_response=True
        )

        # Send and wait for response
        response = await self.send_message(
            message,
            wait_for_response=True,
            timeout=timeout
        )

        if response:
            return response.payload
        else:
            return {
                'success': False,
                'error': 'Timeout or no response'
            }


# Global MCP handler instance
mcp_handler = MCPProtocolHandler()
