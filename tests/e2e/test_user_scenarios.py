"""
End-to-end user scenario tests.

These tests simulate real user interactions with JARVIS.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.agents.jarvis_agent import JarvisAgent
from src.core.memory import MemoryManager
from src.core.llm import LLMProvider


@pytest.fixture
async def jarvis_system():
    """Set up complete JARVIS system for testing."""
    # Mock dependencies
    llm = Mock(spec=LLMProvider)
    llm.generate = AsyncMock(return_value="This is a mock response from the AI assistant.")

    memory = Mock(spec=MemoryManager)
    memory.add_to_short_term = Mock()
    memory.add_to_long_term = Mock()
    memory.query_long_term = Mock(return_value={"documents": [[]]})

    # Create JARVIS instance
    jarvis = JarvisAgent(memory=memory, llm=llm)

    yield jarvis

    # Cleanup
    jarvis.clear_conversation()


class TestMorningRoutine:
    """Test morning routine scenario."""

    @pytest.mark.asyncio
    async def test_morning_briefing(self, jarvis_system):
        """
        Scenario: User asks for morning briefing.

        Expected: JARVIS provides calendar, news, weather summary.
        """
        result = await jarvis_system.execute({
            "query": "Good morning JARVIS, what's on my schedule today?",
            "context": {}
        })

        assert result["success"] is True
        assert "response" in result
        assert result["metadata"]["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_check_emails(self, jarvis_system):
        """
        Scenario: User asks to check important emails.

        Expected: JARVIS summarizes important emails.
        """
        result = await jarvis_system.execute({
            "query": "Check my emails and tell me if there's anything urgent",
            "context": {}
        })

        assert result["success"] is True
        # Should route to email agent
        assert "email" in str(result).lower() or "message" in str(result).lower()


class TestWorkProductivity:
    """Test work productivity scenarios."""

    @pytest.mark.asyncio
    async def test_schedule_meeting(self, jarvis_system):
        """
        Scenario: User wants to schedule a meeting.

        Expected: JARVIS finds available slots and schedules.
        """
        result = await jarvis_system.execute({
            "query": "Schedule a 1-hour meeting with the team tomorrow afternoon",
            "context": {
                "user_timezone": "America/New_York"
            }
        })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_research_topic(self, jarvis_system):
        """
        Scenario: User needs research on a topic.

        Expected: JARVIS conducts research and provides summary.
        """
        result = await jarvis_system.execute({
            "query": "Research the latest developments in quantum computing",
            "context": {}
        })

        assert result["success"] is True
        # Should use researcher agent
        assert result["metadata"]["agent_used"] in ["researcher", "Researcher Agent"]

    @pytest.mark.asyncio
    async def test_draft_email_response(self, jarvis_system):
        """
        Scenario: User needs help drafting an email.

        Expected: JARVIS generates appropriate draft.
        """
        result = await jarvis_system.execute({
            "query": "Draft a professional email declining a meeting invitation",
            "context": {}
        })

        assert result["success"] is True


class TestDataAnalysis:
    """Test data analysis scenarios."""

    @pytest.mark.asyncio
    async def test_analyze_data(self, jarvis_system):
        """
        Scenario: User provides data for analysis.

        Expected: JARVIS analyzes and provides insights.
        """
        result = await jarvis_system.execute({
            "query": "Analyze this sales data and tell me the trends",
            "context": {
                "data": {
                    "sales": [100, 120, 115, 140, 160, 155, 180],
                    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
                }
            }
        })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_visualization(self, jarvis_system):
        """
        Scenario: User wants a data visualization.

        Expected: JARVIS generates visualization spec.
        """
        result = await jarvis_system.execute({
            "query": "Create a chart showing the sales data",
            "context": {
                "data": [100, 150, 200, 175, 225]
            }
        })

        assert result["success"] is True


class TestSmartHome:
    """Test smart home control scenarios."""

    @pytest.mark.asyncio
    async def test_control_lights(self, jarvis_system):
        """
        Scenario: User wants to control lights.

        Expected: JARVIS controls the lights.
        """
        result = await jarvis_system.execute({
            "query": "Turn on the living room lights at 75% brightness",
            "context": {}
        })

        assert result["success"] is True
        # Should route to IoT controller
        assert result["metadata"]["agent_used"] in ["iot_controller", "IoT Controller Agent"]

    @pytest.mark.asyncio
    async def test_temperature_control(self, jarvis_system):
        """
        Scenario: User adjusts temperature.

        Expected: JARVIS adjusts thermostat.
        """
        result = await jarvis_system.execute({
            "query": "Set the bedroom temperature to 22 degrees",
            "context": {}
        })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_security_status(self, jarvis_system):
        """
        Scenario: User checks security status.

        Expected: JARVIS reports security system status.
        """
        result = await jarvis_system.execute({
            "query": "Check the security status of my home",
            "context": {}
        })

        assert result["success"] is True


class TestComplexMultiStep:
    """Test complex multi-step scenarios."""

    @pytest.mark.asyncio
    async def test_plan_event(self, jarvis_system):
        """
        Scenario: User wants to plan a comprehensive event.

        Expected: JARVIS orchestrates multiple agents.
        Steps:
        1. Research venue options
        2. Check calendar availability
        3. Draft invitations
        4. Create task list
        """
        result = await jarvis_system.execute({
            "query": "Help me plan a team offsite event next month",
            "context": {}
        })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_travel_preparation(self, jarvis_system):
        """
        Scenario: User preparing for travel.

        Expected: JARVIS coordinates multiple tasks.
        Steps:
        1. Check calendar conflicts
        2. Research destination
        3. Set home automation
        4. Draft out-of-office message
        """
        result = await jarvis_system.execute({
            "query": "I'm traveling to San Francisco next week, help me prepare",
            "context": {}
        })

        assert result["success"] is True


class TestConversationalMemory:
    """Test conversational context and memory."""

    @pytest.mark.asyncio
    async def test_context_retention(self, jarvis_system):
        """
        Scenario: Multi-turn conversation with context.

        Expected: JARVIS maintains context across turns.
        """
        # First query
        result1 = await jarvis_system.execute({
            "query": "What's the weather like today?",
            "context": {}
        })

        assert result1["success"] is True

        # Follow-up query (should maintain context)
        result2 = await jarvis_system.execute({
            "query": "What about tomorrow?",
            "context": {}
        })

        assert result2["success"] is True

        # Check conversation history
        history = jarvis_system.get_conversation_history()
        assert len(history) >= 4  # 2 queries + 2 responses

    @pytest.mark.asyncio
    async def test_preference_learning(self, jarvis_system):
        """
        Scenario: JARVIS learns user preferences.

        Expected: Preferences are stored and recalled.
        """
        # Express preference
        result1 = await jarvis_system.execute({
            "query": "I prefer meetings in the afternoon",
            "context": {}
        })

        assert result1["success"] is True

        # Later query should consider preference
        result2 = await jarvis_system.execute({
            "query": "Schedule a meeting with the team",
            "context": {}
        })

        assert result2["success"] is True


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_ambiguous_request(self, jarvis_system):
        """
        Scenario: User provides ambiguous request.

        Expected: JARVIS asks for clarification.
        """
        result = await jarvis_system.execute({
            "query": "Do that thing",
            "context": {}
        })

        assert result["success"] is True
        # Should ask for clarification

    @pytest.mark.asyncio
    async def test_impossible_request(self, jarvis_system):
        """
        Scenario: User requests something impossible.

        Expected: JARVIS politely explains limitations.
        """
        result = await jarvis_system.execute({
            "query": "Book a flight to Mars for tomorrow",
            "context": {}
        })

        assert result["success"] is True
        # Should handle gracefully

    @pytest.mark.asyncio
    async def test_conflicting_requests(self, jarvis_system):
        """
        Scenario: User makes conflicting requests.

        Expected: JARVIS identifies conflict and seeks clarification.
        """
        # First request
        result1 = await jarvis_system.execute({
            "query": "Schedule a meeting at 2 PM",
            "context": {}
        })

        # Conflicting request
        result2 = await jarvis_system.execute({
            "query": "Actually, make that 3 PM instead",
            "context": {}
        })

        assert result2["success"] is True


class TestProactiveAssistance:
    """Test proactive assistance scenarios."""

    @pytest.mark.asyncio
    async def test_reminder_suggestions(self, jarvis_system):
        """
        Scenario: JARVIS proactively suggests reminders.

        Expected: Identifies tasks needing reminders.
        """
        result = await jarvis_system.execute({
            "query": "I need to submit the report by Friday",
            "context": {}
        })

        assert result["success"] is True
        # Should offer to set reminder

    @pytest.mark.asyncio
    async def test_pattern_detection(self, jarvis_system):
        """
        Scenario: JARVIS detects patterns in behavior.

        Expected: Offers insights based on patterns.
        """
        # Simulate multiple similar requests
        for i in range(3):
            await jarvis_system.execute({
                "query": f"Check my calendar for meeting {i}",
                "context": {}
            })

        # JARVIS should notice pattern
        result = await jarvis_system.execute({
            "query": "What have I been asking about?",
            "context": {}
        })

        assert result["success"] is True


class TestPerformanceMetrics:
    """Test performance and quality metrics."""

    @pytest.mark.asyncio
    async def test_response_time(self, jarvis_system):
        """Test that responses are timely."""
        result = await jarvis_system.execute({
            "query": "What time is it?",
            "context": {}
        })

        assert result["success"] is True
        # Should respond quickly
        assert result["metadata"]["execution_time"] < 10.0  # seconds

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, jarvis_system):
        """Test handling multiple concurrent requests."""
        # Create multiple tasks
        tasks = [
            jarvis_system.execute({"query": f"Query {i}", "context": {}})
            for i in range(5)
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r["success"] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
