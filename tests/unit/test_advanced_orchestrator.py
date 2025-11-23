"""
Unit tests for Advanced Orchestrator
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.core.advanced_orchestrator import (
    AgentProfile,
    AgentCapability,
    Task,
    TaskPriority,
    TaskStatus,
    AgentResponse,
    ConsensusEngine,
    ConsensusStrategy,
    AdvancedOrchestrator
)


class TestAgentCapability:
    """Test AgentCapability class"""

    def test_initialization(self):
        """Test capability initialization"""
        capability = AgentCapability(
            capability_name="web_search",
            expertise_level=0.9,
            avg_latency_ms=500.0,
            success_rate=0.95
        )

        assert capability.capability_name == "web_search"
        assert capability.expertise_level == 0.9
        assert capability.avg_latency_ms == 500.0
        assert capability.success_rate == 0.95


class TestAgentProfile:
    """Test AgentProfile class"""

    def test_initialization(self):
        """Test profile initialization"""
        capabilities = [
            AgentCapability("search", 0.9),
            AgentCapability("analysis", 0.8)
        ]

        profile = AgentProfile(
            agent_id="agent_1",
            agent_type="researcher",
            capabilities=capabilities
        )

        assert profile.agent_id == "agent_1"
        assert profile.agent_type == "researcher"
        assert len(profile.capabilities) == 2

    def test_has_capability(self):
        """Test capability checking"""
        capabilities = [AgentCapability("search", 0.9)]
        profile = AgentProfile("agent_1", "researcher", capabilities)

        assert profile.has_capability("search")
        assert not profile.has_capability("analysis")

    def test_get_expertise(self):
        """Test getting expertise level"""
        capabilities = [AgentCapability("search", 0.9)]
        profile = AgentProfile("agent_1", "researcher", capabilities)

        assert profile.get_expertise("search") == 0.9
        assert profile.get_expertise("unknown") == 0.0

    def test_availability(self):
        """Test availability checking"""
        profile = AgentProfile("agent_1", "test", [])
        profile.max_load = 3

        assert profile.is_available()

        profile.current_load = 3
        assert not profile.is_available()

    def test_success_rate(self):
        """Test success rate calculation"""
        profile = AgentProfile("agent_1", "test", [])

        profile.total_tasks_completed = 8
        profile.total_tasks_failed = 2

        assert profile.get_success_rate() == 0.8


class TestTask:
    """Test Task class"""

    def test_initialization(self):
        """Test task initialization"""
        task = Task(
            task_id="task_1",
            description="Test task",
            task_type="research",
            priority=TaskPriority.HIGH,
            parameters={"query": "test"}
        )

        assert task.task_id == "task_1"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.HIGH

    def test_assign_agent(self):
        """Test assigning agent to task"""
        task = Task("task_1", "Test", "test", TaskPriority.NORMAL)

        task.assign_agent("agent_1")

        assert "agent_1" in task.assigned_agents
        assert task.status == TaskStatus.ASSIGNED

    def test_complete_task(self):
        """Test completing a task"""
        task = Task("task_1", "Test", "test", TaskPriority.NORMAL)

        result = {"success": True, "data": "result"}
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None

    def test_fail_task(self):
        """Test failing a task"""
        task = Task("task_1", "Test", "test", TaskPriority.NORMAL)

        task.fail("Error message")

        assert task.status == TaskStatus.FAILED
        assert task.error == "Error message"


class TestAgentResponse:
    """Test AgentResponse class"""

    def test_initialization(self):
        """Test response initialization"""
        response = AgentResponse(
            agent_id="agent_1",
            result={"data": "test"},
            confidence=0.9,
            reasoning="Based on analysis"
        )

        assert response.agent_id == "agent_1"
        assert response.confidence == 0.9
        assert response.reasoning == "Based on analysis"

    def test_to_dict(self):
        """Test converting to dictionary"""
        response = AgentResponse(
            agent_id="agent_1",
            result={"data": "test"},
            confidence=0.85
        )

        result_dict = response.to_dict()

        assert result_dict["agent_id"] == "agent_1"
        assert result_dict["confidence"] == 0.85
        assert "timestamp" in result_dict


class TestConsensusEngine:
    """Test ConsensusEngine class"""

    @pytest.mark.asyncio
    async def test_majority_vote(self):
        """Test majority voting consensus"""
        engine = ConsensusEngine()

        responses = [
            AgentResponse("agent_1", {"answer": "A"}, 0.8),
            AgentResponse("agent_2", {"answer": "A"}, 0.9),
            AgentResponse("agent_3", {"answer": "B"}, 0.7)
        ]

        result, confidence, reasoning = await engine.reach_consensus(
            responses,
            ConsensusStrategy.MAJORITY_VOTE
        )

        assert result["answer"] == "A"
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_best_confidence(self):
        """Test best confidence strategy"""
        engine = ConsensusEngine()

        responses = [
            AgentResponse("agent_1", {"answer": "A"}, 0.7),
            AgentResponse("agent_2", {"answer": "B"}, 0.95),
            AgentResponse("agent_3", {"answer": "C"}, 0.6)
        ]

        result, confidence, reasoning = await engine.reach_consensus(
            responses,
            ConsensusStrategy.BEST_CONFIDENCE
        )

        assert result["answer"] == "B"
        assert confidence == 0.95

    @pytest.mark.asyncio
    async def test_unanimous(self):
        """Test unanimous consensus"""
        engine = ConsensusEngine()

        # All agree
        responses_agree = [
            AgentResponse("agent_1", {"answer": "A"}, 0.8),
            AgentResponse("agent_2", {"answer": "A"}, 0.9),
            AgentResponse("agent_3", {"answer": "A"}, 0.85)
        ]

        result, confidence, reasoning = await engine.reach_consensus(
            responses_agree,
            ConsensusStrategy.UNANIMOUS
        )

        assert result["answer"] == "A"
        assert confidence > 0.5

        # Disagreement
        responses_disagree = [
            AgentResponse("agent_1", {"answer": "A"}, 0.8),
            AgentResponse("agent_2", {"answer": "B"}, 0.9)
        ]

        result, confidence, reasoning = await engine.reach_consensus(
            responses_disagree,
            ConsensusStrategy.UNANIMOUS
        )

        assert result is None
        assert confidence == 0.0


class TestAdvancedOrchestrator:
    """Test AdvancedOrchestrator class"""

    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = AdvancedOrchestrator()

        assert len(orchestrator.agent_profiles) == 0
        assert len(orchestrator.task_queue) == 0
        assert orchestrator.consensus_engine is not None

    def test_register_agent(self):
        """Test registering an agent"""
        orchestrator = AdvancedOrchestrator()

        profile = AgentProfile(
            "agent_1",
            "researcher",
            [AgentCapability("search", 0.9)]
        )

        orchestrator.register_agent(profile)

        assert "agent_1" in orchestrator.agent_profiles

    def test_create_task(self):
        """Test creating a task"""
        orchestrator = AdvancedOrchestrator()

        task = orchestrator.create_task(
            description="Test task",
            task_type="research",
            priority=TaskPriority.HIGH
        )

        assert task.task_id is not None
        assert len(orchestrator.task_queue) == 1

    def test_select_agents_for_task(self):
        """Test agent selection for task"""
        orchestrator = AdvancedOrchestrator()

        # Register agents
        orchestrator.register_agent(AgentProfile(
            "agent_1",
            "researcher",
            [AgentCapability("search", 0.9)]
        ))

        orchestrator.register_agent(AgentProfile(
            "agent_2",
            "analyst",
            [AgentCapability("analysis", 0.8)]
        ))

        # Create task requiring search
        task = Task(
            "task_1",
            "Search task",
            "research",
            TaskPriority.NORMAL,
            required_capabilities=["search"]
        )

        selected = orchestrator._select_agents_for_task(task, max_agents=1)

        assert len(selected) > 0
        assert "agent_1" in selected  # Should select agent with search capability

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test executing a task"""
        orchestrator = AdvancedOrchestrator()

        # Register agent
        orchestrator.register_agent(AgentProfile(
            "agent_1",
            "test",
            [AgentCapability("test", 0.9)]
        ))

        # Create task
        task = orchestrator.create_task(
            description="Test task",
            task_type="test",
            parameters={"input": "test"}
        )

        # Mock executor
        async def mock_executor(params):
            return {"result": "success", "confidence": 0.9}

        executors = {"agent_1": mock_executor}

        # Execute
        result = await orchestrator.execute_task(
            task,
            executors,
            ConsensusStrategy.BEST_CONFIDENCE
        )

        assert result["success"]
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_with_multiple_agents(self):
        """Test executing task with multiple agents"""
        orchestrator = AdvancedOrchestrator()

        # Register multiple agents
        for i in range(3):
            orchestrator.register_agent(AgentProfile(
                f"agent_{i}",
                "test",
                [AgentCapability("test", 0.8 + i * 0.05)]
            ))

        # Create task
        task = orchestrator.create_task(
            description="Multi-agent task",
            task_type="test",
            required_capabilities=["test"]
        )

        # Mock executors
        async def mock_executor(params):
            return {"result": "success", "confidence": 0.85}

        executors = {
            f"agent_{i}": mock_executor
            for i in range(3)
        }

        # Execute with multiple agents
        result = await orchestrator.execute_task(
            task,
            executors,
            ConsensusStrategy.MAJORITY_VOTE,
            require_multiple_agents=True
        )

        assert result["success"]
        assert len(result["agents_used"]) > 1

    def test_get_agent_statistics(self):
        """Test getting agent statistics"""
        orchestrator = AdvancedOrchestrator()

        profile = AgentProfile(
            "agent_1",
            "test",
            [AgentCapability("test", 0.9)]
        )
        profile.total_tasks_completed = 10
        profile.total_tasks_failed = 2

        orchestrator.register_agent(profile)

        stats = orchestrator.get_agent_statistics()

        assert "agent_1" in stats
        assert stats["agent_1"]["tasks_completed"] == 10
        assert stats["agent_1"]["success_rate"] > 0.5

    def test_get_task_statistics(self):
        """Test getting task statistics"""
        orchestrator = AdvancedOrchestrator()

        # Create some tasks
        task1 = Task("task_1", "Test 1", "test", TaskPriority.NORMAL)
        task1.complete({"success": True})
        orchestrator.completed_tasks.append(task1)

        task2 = Task("task_2", "Test 2", "test", TaskPriority.NORMAL)
        task2.fail("Error")
        orchestrator.completed_tasks.append(task2)

        stats = orchestrator.get_task_statistics()

        assert stats["total_tasks"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.5


@pytest.fixture
def orchestrator_with_agents():
    """Fixture providing orchestrator with registered agents"""
    orchestrator = AdvancedOrchestrator()

    # Register multiple agents
    agent_types = ["researcher", "analyst", "controller"]
    capabilities = [
        [AgentCapability("search", 0.9), AgentCapability("fact_check", 0.8)],
        [AgentCapability("analysis", 0.95), AgentCapability("visualization", 0.85)],
        [AgentCapability("device_control", 0.9)]
    ]

    for i, (agent_type, caps) in enumerate(zip(agent_types, capabilities)):
        orchestrator.register_agent(AgentProfile(
            f"agent_{i}",
            agent_type,
            caps
        ))

    return orchestrator


def test_with_orchestrator_fixture(orchestrator_with_agents):
    """Test using orchestrator fixture"""
    assert len(orchestrator_with_agents.agent_profiles) == 3

    stats = orchestrator_with_agents.get_agent_statistics()
    assert len(stats) == 3
