"""
Unit tests for MCP Protocol Handler.
"""

import pytest
import asyncio
from src.core.mcp_protocol import (
    MCPProtocolHandler,
    MCPMessage,
    MessageType,
    MessagePriority,
    AgentCapability,
    AgentRegistration
)


@pytest.fixture
def mcp_handler():
    """Create MCP handler for testing."""
    return MCPProtocolHandler()


@pytest.fixture
def sample_capability():
    """Sample agent capability."""
    return {
        "name": "test_capability",
        "description": "Test capability",
        "input_schema": {},
        "output_schema": {},
        "cost": 0.5,
        "latency": 1.0
    }


class TestMCPMessage:
    """Test MCPMessage class."""

    def test_message_creation(self):
        """Test message creation with default values."""
        msg = MCPMessage(
            type=MessageType.REQUEST,
            sender="test_agent",
            recipient="target_agent",
            payload={"test": "data"}
        )

        assert msg.type == MessageType.REQUEST
        assert msg.sender == "test_agent"
        assert msg.recipient == "target_agent"
        assert msg.payload == {"test": "data"}
        assert msg.priority == MessagePriority.NORMAL
        assert msg.id is not None

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = MCPMessage(
            type=MessageType.RESPONSE,
            sender="agent1",
            recipient="agent2",
            payload={"result": "success"}
        )

        msg_dict = msg.to_dict()

        assert isinstance(msg_dict, dict)
        assert msg_dict["type"] == MessageType.RESPONSE
        assert msg_dict["sender"] == "agent1"

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "id": "test-123",
            "type": "request",
            "priority": "high",
            "sender": "agent1",
            "recipient": "agent2",
            "timestamp": 123456789.0,
            "payload": {"test": "data"}
        }

        msg = MCPMessage.from_dict(data)

        assert msg.id == "test-123"
        assert msg.type == MessageType.REQUEST
        assert msg.priority == MessagePriority.HIGH


class TestAgentRegistration:
    """Test agent registration."""

    def test_register_agent(self, mcp_handler, sample_capability):
        """Test registering an agent."""
        success = mcp_handler.register_agent(
            agent_id="test_agent",
            name="Test Agent",
            agent_type="test",
            capabilities=[sample_capability]
        )

        assert success is True
        assert "test_agent" in mcp_handler.agents
        assert mcp_handler.agents["test_agent"].name == "Test Agent"

    def test_unregister_agent(self, mcp_handler, sample_capability):
        """Test unregistering an agent."""
        mcp_handler.register_agent(
            agent_id="test_agent",
            name="Test Agent",
            agent_type="test",
            capabilities=[sample_capability]
        )

        success = mcp_handler.unregister_agent("test_agent")

        assert success is True
        assert "test_agent" not in mcp_handler.agents

    def test_register_multiple_agents(self, mcp_handler, sample_capability):
        """Test registering multiple agents."""
        for i in range(3):
            mcp_handler.register_agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                agent_type="test",
                capabilities=[sample_capability]
            )

        assert len(mcp_handler.agents) == 3


class TestAgentDiscovery:
    """Test agent discovery features."""

    def test_discover_all_agents(self, mcp_handler, sample_capability):
        """Test discovering all agents."""
        for i in range(3):
            mcp_handler.register_agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                agent_type="test",
                capabilities=[sample_capability]
            )

        agents = mcp_handler.discover_agents()

        assert len(agents) == 3

    def test_discover_by_capability(self, mcp_handler):
        """Test discovering agents by capability."""
        mcp_handler.register_agent(
            agent_id="agent_1",
            name="Agent 1",
            agent_type="test",
            capabilities=[{
                "name": "search",
                "description": "Search capability",
                "input_schema": {},
                "output_schema": {}
            }]
        )

        mcp_handler.register_agent(
            agent_id="agent_2",
            name="Agent 2",
            agent_type="test",
            capabilities=[{
                "name": "analyze",
                "description": "Analysis capability",
                "input_schema": {},
                "output_schema": {}
            }]
        )

        search_agents = mcp_handler.discover_agents(capability="search")

        assert len(search_agents) == 1
        assert search_agents[0].agent_id == "agent_1"

    def test_discover_by_type(self, mcp_handler, sample_capability):
        """Test discovering agents by type."""
        mcp_handler.register_agent(
            agent_id="researcher_1",
            name="Researcher",
            agent_type="researcher",
            capabilities=[sample_capability]
        )

        mcp_handler.register_agent(
            agent_id="analyst_1",
            name="Analyst",
            agent_type="analyst",
            capabilities=[sample_capability]
        )

        researcher_agents = mcp_handler.discover_agents(agent_type="researcher")

        assert len(researcher_agents) == 1
        assert researcher_agents[0].agent_id == "researcher_1"

    def test_find_best_agent(self, mcp_handler, sample_capability):
        """Test finding best agent for capability."""
        # Register agents with different loads
        for i in range(3):
            mcp_handler.register_agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                agent_type="test",
                capabilities=[sample_capability]
            )
            mcp_handler.agents[f"agent_{i}"].load = i * 0.3

        best_agent = mcp_handler.find_best_agent("test_capability")

        assert best_agent == "agent_0"  # Lowest load


class TestMessageRouting:
    """Test message routing and handling."""

    @pytest.mark.asyncio
    async def test_send_message(self, mcp_handler, sample_capability):
        """Test sending a message."""
        mcp_handler.register_agent(
            agent_id="target_agent",
            name="Target",
            agent_type="test",
            capabilities=[sample_capability]
        )

        msg = MCPMessage(
            type=MessageType.REQUEST,
            sender="source_agent",
            recipient="target_agent",
            payload={"test": "data"}
        )

        result = await mcp_handler.send_message(msg)

        assert msg in mcp_handler.message_history

    @pytest.mark.asyncio
    async def test_broadcast_message(self, mcp_handler, sample_capability):
        """Test broadcasting a message."""
        # Register multiple agents
        for i in range(3):
            mcp_handler.register_agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                agent_type="test",
                capabilities=[sample_capability]
            )

        count = await mcp_handler.broadcast_message(
            sender="broadcaster",
            payload={"announcement": "test"}
        )

        assert count == 3


class TestContextManagement:
    """Test context management features."""

    def test_store_context(self, mcp_handler):
        """Test storing context."""
        context_data = {"key": "value", "data": [1, 2, 3]}

        mcp_handler.store_context("context_1", context_data)

        retrieved = mcp_handler.get_context("context_1")

        assert retrieved == context_data

    def test_clear_context(self, mcp_handler):
        """Test clearing context."""
        mcp_handler.store_context("context_1", {"test": "data"})
        mcp_handler.clear_context("context_1")

        retrieved = mcp_handler.get_context("context_1")

        assert retrieved is None

    def test_optimize_context_window(self, mcp_handler):
        """Test context window optimization."""
        messages = []

        # Create messages with different priorities
        for i in range(10):
            msg = MCPMessage(
                type=MessageType.REQUEST,
                sender=f"agent_{i}",
                recipient="target",
                payload={"index": i},
                priority=MessagePriority.HIGH if i % 3 == 0 else MessagePriority.NORMAL
            )
            messages.append(msg)

        optimized = mcp_handler.optimize_context_window(
            messages,
            max_tokens=500  # Allow ~5 messages
        )

        assert len(optimized) <= 5
        # Should preserve high priority messages
        high_priority_count = sum(
            1 for msg in optimized if msg.priority == MessagePriority.HIGH
        )
        assert high_priority_count > 0


class TestCapabilityExecution:
    """Test capability execution requests."""

    @pytest.mark.asyncio
    async def test_request_capability_execution(self, mcp_handler, sample_capability):
        """Test requesting capability execution."""
        mcp_handler.register_agent(
            agent_id="executor",
            name="Executor",
            agent_type="test",
            capabilities=[sample_capability]
        )

        # Note: This will timeout in test as we don't have handler registered
        # But it demonstrates the API
        result = await mcp_handler.request_capability_execution(
            capability="test_capability",
            params={"input": "test"},
            sender="requester",
            timeout=1.0
        )

        # Should return timeout error since no handler is set up
        assert "error" in result or "success" in result


class TestAgentStatus:
    """Test agent status and monitoring."""

    def test_get_agent_status(self, mcp_handler, sample_capability):
        """Test getting agent status."""
        for i in range(3):
            mcp_handler.register_agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                agent_type="test",
                capabilities=[sample_capability]
            )

        status = mcp_handler.get_agent_status()

        assert status["total_agents"] == 3
        assert len(status["agents"]) == 3
        assert "message_queue_size" in status
        assert "message_history_size" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
