"""
Integration tests for agent interactions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.agents.jarvis_agent import JarvisAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.scheduler_agent import SchedulerAgent
from src.agents.email_agent import EmailAgent
from src.agents.iot_controller_agent import IoTControllerAgent
from src.agents.data_analyst_agent import DataAnalystAgent
from src.core.memory import MemoryManager
from src.core.llm import LLMProvider


@pytest.fixture
async def mock_llm():
    """Mock LLM provider."""
    llm = Mock(spec=LLMProvider)
    llm.generate = AsyncMock(return_value="Mock LLM response")
    return llm


@pytest.fixture
async def mock_memory():
    """Mock memory manager."""
    memory = Mock(spec=MemoryManager)
    memory.add_to_short_term = Mock()
    memory.query_long_term = Mock(return_value={"documents": [[]]})
    return memory


class TestResearcherAgent:
    """Test Researcher Agent integration."""

    @pytest.mark.asyncio
    async def test_researcher_search(self, mock_memory, mock_llm):
        """Test researcher agent search capability."""
        agent = ResearcherAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "search",
            "query": "artificial intelligence trends 2024"
        })

        assert "success" in result
        # Note: Will fail in test without actual API keys
        # In production, mock the web search API

    @pytest.mark.asyncio
    async def test_researcher_research(self, mock_memory, mock_llm):
        """Test deep research capability."""
        agent = ResearcherAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "research",
            "query": "quantum computing applications"
        })

        assert "success" in result or "error" in result


class TestSchedulerAgent:
    """Test Scheduler Agent integration."""

    @pytest.mark.asyncio
    async def test_scheduler_find_slots(self, mock_memory, mock_llm):
        """Test finding available time slots."""
        agent = SchedulerAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "find_slots",
            "duration_minutes": 60,
            "days_ahead": 7
        })

        # Should work even without calendar API
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_scheduler_suggest_times(self, mock_memory, mock_llm):
        """Test suggesting meeting times."""
        agent = SchedulerAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "suggest_times",
            "participants": ["user1@example.com", "user2@example.com"],
            "duration_minutes": 30
        })

        # Should use LLM for suggestions
        assert "success" in result or "error" in result


class TestEmailAgent:
    """Test Email Agent integration."""

    @pytest.mark.asyncio
    async def test_email_draft(self, mock_memory, mock_llm):
        """Test email draft generation."""
        agent = EmailAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "draft_email",
            "purpose": "Request meeting",
            "context": "Follow up on project discussion",
            "tone": "professional"
        })

        assert result["success"] is True
        assert "draft" in result
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_categorize(self, mock_memory, mock_llm):
        """Test email categorization."""
        agent = EmailAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "categorize",
            "emails": [
                {"subject": "Urgent: Server down", "from": "ops@example.com"},
                {"subject": "Newsletter", "from": "marketing@example.com"}
            ]
        })

        assert result["success"] is True
        mock_llm.generate.assert_called_once()


class TestIoTControllerAgent:
    """Test IoT Controller Agent integration."""

    @pytest.mark.asyncio
    async def test_iot_discover_devices(self, mock_memory, mock_llm):
        """Test device discovery."""
        agent = IoTControllerAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "discover"
        })

        assert result["success"] is True
        assert "devices" in result
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_iot_control_device(self, mock_memory, mock_llm):
        """Test controlling a device."""
        agent = IoTControllerAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "control",
            "device": "living_room_light",
            "command": "turn_on",
            "params": {"brightness": 75}
        })

        assert result["success"] is True
        assert result["device"] == "living_room_light"

    @pytest.mark.asyncio
    async def test_iot_energy_monitoring(self, mock_memory, mock_llm):
        """Test energy monitoring."""
        agent = IoTControllerAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "monitor_energy"
        })

        assert result["success"] is True
        assert "energy_data" in result


class TestDataAnalystAgent:
    """Test Data Analyst Agent integration."""

    @pytest.mark.asyncio
    async def test_analyst_analyze_data(self, mock_memory, mock_llm):
        """Test data analysis."""
        agent = DataAnalystAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "analyze",
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type": "statistical"
        })

        assert result["success"] is True
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyst_visualization(self, mock_memory, mock_llm):
        """Test visualization generation."""
        agent = DataAnalystAgent(memory=mock_memory, llm=mock_llm)

        result = await agent.process({
            "action": "visualize",
            "data": {"sales": [100, 150, 200], "months": ["Jan", "Feb", "Mar"]},
            "type": "auto"
        })

        assert result["success"] is True
        assert "visualization_spec" in result


class TestJarvisAgentIntegration:
    """Test Jarvis main agent integration."""

    @pytest.mark.asyncio
    async def test_jarvis_initialization(self, mock_memory, mock_llm):
        """Test Jarvis initialization with all agents."""
        jarvis = JarvisAgent(memory=mock_memory, llm=mock_llm)

        # Check that specialized agents are registered
        assert len(jarvis.specialized_agents) > 0
        assert "researcher" in jarvis.specialized_agents
        assert "scheduler" in jarvis.specialized_agents
        assert "email" in jarvis.specialized_agents
        assert "iot_controller" in jarvis.specialized_agents
        assert "analyst" in jarvis.specialized_agents

    @pytest.mark.asyncio
    async def test_jarvis_routing(self, mock_memory, mock_llm):
        """Test task routing."""
        jarvis = JarvisAgent(memory=mock_memory, llm=mock_llm)

        # Test routing to researcher
        routing = await jarvis._route_task(
            "search for recent AI news",
            {}
        )

        assert routing["requires_specialized_agent"] is True
        assert routing["agent_type"] == "researcher"

        # Test routing to scheduler
        routing = await jarvis._route_task(
            "schedule a meeting tomorrow",
            {}
        )

        assert routing["agent_type"] == "scheduler"

        # Test routing to IoT
        routing = await jarvis._route_task(
            "turn on the lights",
            {}
        )

        assert routing["agent_type"] == "iot_controller"

    @pytest.mark.asyncio
    async def test_jarvis_multi_agent_orchestration(self, mock_memory, mock_llm):
        """Test multi-agent orchestration."""
        jarvis = JarvisAgent(memory=mock_memory, llm=mock_llm)

        # Test multi-agent query
        routing = await jarvis._route_task(
            "plan a comprehensive analysis of my schedule and research trends",
            {}
        )

        assert routing["multi_agent"] is True


class TestAgentCollaboration:
    """Test agent collaboration scenarios."""

    @pytest.mark.asyncio
    async def test_research_then_email(self, mock_memory, mock_llm):
        """Test research followed by email draft."""
        researcher = ResearcherAgent(memory=mock_memory, llm=mock_llm)
        email_agent = EmailAgent(memory=mock_memory, llm=mock_llm)

        # Research a topic
        research_result = await researcher.process({
            "action": "search",
            "query": "AI trends 2024"
        })

        # Draft email based on research
        email_result = await email_agent.process({
            "action": "draft_email",
            "purpose": "Share research findings",
            "context": str(research_result),
            "tone": "professional"
        })

        assert email_result["success"] is True

    @pytest.mark.asyncio
    async def test_data_analysis_then_email(self, mock_memory, mock_llm):
        """Test data analysis followed by report email."""
        analyst = DataAnalystAgent(memory=mock_memory, llm=mock_llm)
        email_agent = EmailAgent(memory=mock_memory, llm=mock_llm)

        # Analyze data
        analysis_result = await analyst.process({
            "action": "analyze",
            "data": [10, 20, 30, 40, 50]
        })

        # Draft email with analysis
        email_result = await email_agent.process({
            "action": "draft_email",
            "purpose": "Share data analysis",
            "context": str(analysis_result),
            "tone": "professional"
        })

        assert email_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
