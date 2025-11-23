"""
Advanced Multi-Agent Orchestrator for JARVIS
Handles complex task coordination, consensus mechanisms, and intelligent agent collaboration
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum
import asyncio
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus among agents"""
    MAJORITY_VOTE = "majority_vote"  # Simple majority
    WEIGHTED_VOTE = "weighted_vote"  # Based on agent expertise
    UNANIMOUS = "unanimous"  # All agents must agree
    FIRST_SUCCESS = "first_success"  # First successful response
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence score


class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentResponse:
    """Response from an agent"""

    def __init__(
        self,
        agent_id: str,
        result: Any,
        confidence: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.result = result
        self.confidence = confidence
        self.reasoning = reasoning or ""
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "result": self.result,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class Task:
    """Represents a task to be executed"""

    def __init__(
        self,
        task_id: str,
        description: str,
        task_type: str,
        priority: TaskPriority,
        parameters: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[str]] = None,
        deadline: Optional[datetime] = None
    ):
        self.task_id = task_id
        self.description = description
        self.task_type = task_type
        self.priority = priority
        self.parameters = parameters or {}
        self.required_capabilities = required_capabilities or []
        self.deadline = deadline
        self.status = TaskStatus.PENDING
        self.assigned_agents: List[str] = []
        self.responses: List[AgentResponse] = []
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[str] = None

    def assign_agent(self, agent_id: str):
        """Assign an agent to this task"""
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)
        self.status = TaskStatus.ASSIGNED

    def add_response(self, response: AgentResponse):
        """Add agent response"""
        self.responses.append(response)

    def complete(self, result: Any):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_agents": self.assigned_agents,
            "responses": [r.to_dict() for r in self.responses],
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class AgentCapability:
    """Represents an agent's capability"""

    def __init__(
        self,
        capability_name: str,
        expertise_level: float,
        avg_latency_ms: float = 1000.0,
        success_rate: float = 0.95
    ):
        self.capability_name = capability_name
        self.expertise_level = expertise_level  # 0.0 to 1.0
        self.avg_latency_ms = avg_latency_ms
        self.success_rate = success_rate


class AgentProfile:
    """Profile of an agent with capabilities and performance metrics"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability]
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = {cap.capability_name: cap for cap in capabilities}
        self.current_load = 0
        self.max_load = 5
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.average_response_time_ms = 0.0
        self.last_active = datetime.now()

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a capability"""
        return capability in self.capabilities

    def get_expertise(self, capability: str) -> float:
        """Get expertise level for a capability"""
        if capability in self.capabilities:
            return self.capabilities[capability].expertise_level
        return 0.0

    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.current_load < self.max_load

    def get_success_rate(self) -> float:
        """Get overall success rate"""
        total = self.total_tasks_completed + self.total_tasks_failed
        if total == 0:
            return 0.95  # Default
        return self.total_tasks_completed / total


class ConsensusEngine:
    """Engine for reaching consensus among multiple agents"""

    def __init__(self):
        self.strategy_handlers = {
            ConsensusStrategy.MAJORITY_VOTE: self._majority_vote,
            ConsensusStrategy.WEIGHTED_VOTE: self._weighted_vote,
            ConsensusStrategy.UNANIMOUS: self._unanimous,
            ConsensusStrategy.FIRST_SUCCESS: self._first_success,
            ConsensusStrategy.BEST_CONFIDENCE: self._best_confidence
        }

    async def reach_consensus(
        self,
        responses: List[AgentResponse],
        strategy: ConsensusStrategy,
        agent_profiles: Optional[Dict[str, AgentProfile]] = None
    ) -> Tuple[Any, float, str]:
        """
        Reach consensus from multiple agent responses
        Returns: (result, confidence, reasoning)
        """
        if not responses:
            return None, 0.0, "No responses received"

        handler = self.strategy_handlers.get(strategy)
        if not handler:
            return None, 0.0, f"Unknown strategy: {strategy}"

        return await handler(responses, agent_profiles or {})

    async def _majority_vote(
        self,
        responses: List[AgentResponse],
        agent_profiles: Dict[str, AgentProfile]
    ) -> Tuple[Any, float, str]:
        """Simple majority voting"""
        # Group responses by result
        result_groups = defaultdict(list)
        for response in responses:
            result_key = json.dumps(response.result, sort_keys=True)
            result_groups[result_key].append(response)

        # Find majority
        majority_group = max(result_groups.values(), key=len)
        majority_result = majority_group[0].result

        # Calculate confidence
        confidence = len(majority_group) / len(responses)
        avg_response_confidence = sum(r.confidence for r in majority_group) / len(majority_group)
        final_confidence = (confidence + avg_response_confidence) / 2

        reasoning = (
            f"Majority consensus: {len(majority_group)}/{len(responses)} agents agreed. "
            f"Average agent confidence: {avg_response_confidence:.2f}"
        )

        return majority_result, final_confidence, reasoning

    async def _weighted_vote(
        self,
        responses: List[AgentResponse],
        agent_profiles: Dict[str, AgentProfile]
    ) -> Tuple[Any, float, str]:
        """Weighted voting based on agent expertise and confidence"""
        # Calculate weighted scores
        result_scores = defaultdict(float)
        result_responses = defaultdict(list)

        for response in responses:
            result_key = json.dumps(response.result, sort_keys=True)

            # Get agent weight (expertise × success rate)
            agent_profile = agent_profiles.get(response.agent_id)
            if agent_profile:
                agent_weight = agent_profile.get_success_rate()
            else:
                agent_weight = 0.5  # Default weight

            # Combined score
            score = response.confidence * agent_weight
            result_scores[result_key] += score
            result_responses[result_key].append(response)

        # Find highest weighted result
        best_result_key = max(result_scores.items(), key=lambda x: x[1])[0]
        best_result = result_responses[best_result_key][0].result
        best_score = result_scores[best_result_key]

        # Normalize confidence
        total_possible_score = len(responses)
        confidence = min(best_score / total_possible_score, 1.0)

        reasoning = (
            f"Weighted consensus: score {best_score:.2f} from "
            f"{len(result_responses[best_result_key])} agents"
        )

        return best_result, confidence, reasoning

    async def _unanimous(
        self,
        responses: List[AgentResponse],
        agent_profiles: Dict[str, AgentProfile]
    ) -> Tuple[Any, float, str]:
        """Require unanimous agreement"""
        if not responses:
            return None, 0.0, "No responses"

        # Check if all results are the same
        first_result = json.dumps(responses[0].result, sort_keys=True)

        for response in responses[1:]:
            if json.dumps(response.result, sort_keys=True) != first_result:
                return None, 0.0, "No unanimous agreement reached"

        # All agreed
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        reasoning = f"Unanimous agreement from {len(responses)} agents"

        return responses[0].result, avg_confidence, reasoning

    async def _first_success(
        self,
        responses: List[AgentResponse],
        agent_profiles: Dict[str, AgentProfile]
    ) -> Tuple[Any, float, str]:
        """Return first successful response above threshold"""
        threshold = 0.7

        for response in responses:
            if response.confidence >= threshold:
                reasoning = (
                    f"First successful response from {response.agent_id} "
                    f"with confidence {response.confidence:.2f}"
                )
                return response.result, response.confidence, reasoning

        # No response met threshold, return best confidence
        best_response = max(responses, key=lambda r: r.confidence)
        reasoning = (
            f"No response met threshold {threshold}, "
            f"returning best: {best_response.confidence:.2f}"
        )

        return best_response.result, best_response.confidence, reasoning

    async def _best_confidence(
        self,
        responses: List[AgentResponse],
        agent_profiles: Dict[str, AgentProfile]
    ) -> Tuple[Any, float, str]:
        """Return response with highest confidence"""
        best_response = max(responses, key=lambda r: r.confidence)

        reasoning = (
            f"Highest confidence response from {best_response.agent_id}: "
            f"{best_response.confidence:.2f}"
        )

        return best_response.result, best_response.confidence, reasoning


class AdvancedOrchestrator:
    """Advanced orchestrator for multi-agent coordination"""

    def __init__(self):
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.consensus_engine = ConsensusEngine()
        self.task_counter = 0

    def register_agent(self, agent_profile: AgentProfile):
        """Register an agent with the orchestrator"""
        self.agent_profiles[agent_profile.agent_id] = agent_profile
        logger.info(f"Registered agent: {agent_profile.agent_id} ({agent_profile.agent_type})")

    def create_task(
        self,
        description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        parameters: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[str]] = None,
        deadline: Optional[datetime] = None
    ) -> Task:
        """Create a new task"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{int(datetime.now().timestamp())}"

        task = Task(
            task_id=task_id,
            description=description,
            task_type=task_type,
            priority=priority,
            parameters=parameters,
            required_capabilities=required_capabilities,
            deadline=deadline
        )

        self.task_queue.append(task)
        logger.info(f"Created task: {task_id} - {description}")

        return task

    def _select_agents_for_task(
        self,
        task: Task,
        max_agents: int = 3
    ) -> List[str]:
        """Select best agents for a task"""
        candidate_scores = []

        for agent_id, profile in self.agent_profiles.items():
            if not profile.is_available():
                continue

            # Check required capabilities
            if task.required_capabilities:
                if not all(profile.has_capability(cap) for cap in task.required_capabilities):
                    continue

            # Calculate suitability score
            score = 0.0

            # Expertise score
            if task.required_capabilities:
                expertise_scores = [
                    profile.get_expertise(cap)
                    for cap in task.required_capabilities
                ]
                score += sum(expertise_scores) / len(expertise_scores)

            # Success rate
            score += profile.get_success_rate() * 0.5

            # Load factor (prefer less loaded agents)
            load_factor = 1.0 - (profile.current_load / profile.max_load)
            score += load_factor * 0.3

            candidate_scores.append((agent_id, score))

        # Sort by score and select top agents
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [agent_id for agent_id, _ in candidate_scores[:max_agents]]

        return selected

    async def execute_task(
        self,
        task: Task,
        agent_executors: Dict[str, Callable],
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTE,
        require_multiple_agents: bool = False
    ) -> Dict[str, Any]:
        """Execute a task with selected agents"""
        # Select agents
        selected_agents = self._select_agents_for_task(
            task,
            max_agents=3 if require_multiple_agents else 1
        )

        if not selected_agents:
            task.fail("No suitable agents available")
            return {"success": False, "error": "No agents available"}

        # Assign agents
        for agent_id in selected_agents:
            task.assign_agent(agent_id)
            self.agent_profiles[agent_id].current_load += 1

        task.status = TaskStatus.IN_PROGRESS
        self.active_tasks[task.task_id] = task

        logger.info(
            f"Executing task {task.task_id} with agents: {selected_agents}"
        )

        # Execute with all selected agents
        agent_responses = []

        for agent_id in selected_agents:
            try:
                # Get executor for this agent
                executor = agent_executors.get(agent_id)
                if not executor:
                    logger.warning(f"No executor for agent {agent_id}")
                    continue

                # Execute
                start_time = datetime.now()
                result = await executor(task.parameters)
                latency = (datetime.now() - start_time).total_seconds() * 1000

                # Create response (assume confidence from result if available)
                confidence = result.get("confidence", 0.8) if isinstance(result, dict) else 0.8

                response = AgentResponse(
                    agent_id=agent_id,
                    result=result,
                    confidence=confidence,
                    reasoning=result.get("reasoning", "") if isinstance(result, dict) else "",
                    metadata={"latency_ms": latency}
                )

                agent_responses.append(response)
                task.add_response(response)

                # Update agent metrics
                profile = self.agent_profiles[agent_id]
                profile.total_tasks_completed += 1
                profile.average_response_time_ms = (
                    0.9 * profile.average_response_time_ms + 0.1 * latency
                )

            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                profile = self.agent_profiles[agent_id]
                profile.total_tasks_failed += 1

            finally:
                # Reduce load
                self.agent_profiles[agent_id].current_load -= 1

        # Reach consensus if multiple responses
        if len(agent_responses) > 1:
            result, confidence, reasoning = await self.consensus_engine.reach_consensus(
                agent_responses,
                consensus_strategy,
                self.agent_profiles
            )
        elif len(agent_responses) == 1:
            result = agent_responses[0].result
            confidence = agent_responses[0].confidence
            reasoning = agent_responses[0].reasoning
        else:
            task.fail("All agents failed to respond")
            return {"success": False, "error": "All agents failed"}

        # Complete task
        task.complete(result)
        self.completed_tasks.append(task)
        del self.active_tasks[task.task_id]

        return {
            "success": True,
            "result": result,
            "confidence": confidence,
            "reasoning": reasoning,
            "agents_used": selected_agents,
            "consensus_strategy": consensus_strategy.value
        }

    async def execute_complex_workflow(
        self,
        workflow_steps: List[Dict[str, Any]],
        agent_executors: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Execute a complex multi-step workflow"""
        results = []
        context = {}

        for step in workflow_steps:
            # Create task for this step
            task = self.create_task(
                description=step["description"],
                task_type=step["type"],
                priority=TaskPriority(step.get("priority", 3)),
                parameters={**step.get("parameters", {}), **context},
                required_capabilities=step.get("capabilities")
            )

            # Execute task
            result = await self.execute_task(
                task,
                agent_executors,
                consensus_strategy=ConsensusStrategy(
                    step.get("consensus", "weighted_vote")
                ),
                require_multiple_agents=step.get("require_multiple", False)
            )

            results.append({
                "step": step["description"],
                "result": result
            })

            # Update context with result
            if result["success"]:
                context.update(result.get("result", {}))
            else:
                # Workflow failed
                return {
                    "success": False,
                    "failed_at": step["description"],
                    "results": results
                }

        return {
            "success": True,
            "results": results,
            "final_context": context
        }

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about all agents"""
        stats = {}

        for agent_id, profile in self.agent_profiles.items():
            stats[agent_id] = {
                "type": profile.agent_type,
                "capabilities": list(profile.capabilities.keys()),
                "current_load": profile.current_load,
                "max_load": profile.max_load,
                "tasks_completed": profile.total_tasks_completed,
                "tasks_failed": profile.total_tasks_failed,
                "success_rate": profile.get_success_rate(),
                "avg_response_time_ms": profile.average_response_time_ms
            }

        return stats

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about tasks"""
        total = len(self.completed_tasks)
        successful = sum(1 for t in self.completed_tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.completed_tasks if t.status == TaskStatus.FAILED)

        return {
            "total_tasks": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue)
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Create orchestrator
        orchestrator = AdvancedOrchestrator()

        # Register agents
        orchestrator.register_agent(AgentProfile(
            agent_id="researcher_1",
            agent_type="researcher",
            capabilities=[
                AgentCapability("web_search", 0.9),
                AgentCapability("fact_checking", 0.85)
            ]
        ))

        orchestrator.register_agent(AgentProfile(
            agent_id="analyst_1",
            agent_type="analyst",
            capabilities=[
                AgentCapability("data_analysis", 0.95),
                AgentCapability("visualization", 0.8)
            ]
        ))

        # Create mock executors
        async def mock_executor(params):
            await asyncio.sleep(0.1)
            return {"result": f"Processed {params}", "confidence": 0.9}

        executors = {
            "researcher_1": mock_executor,
            "analyst_1": mock_executor
        }

        # Create and execute task
        task = orchestrator.create_task(
            description="Research and analyze topic",
            task_type="research",
            priority=TaskPriority.HIGH,
            parameters={"topic": "AI trends"},
            required_capabilities=["web_search"]
        )

        result = await orchestrator.execute_task(
            task,
            executors,
            consensus_strategy=ConsensusStrategy.BEST_CONFIDENCE
        )

        print("Task result:")
        print(json.dumps(result, indent=2, default=str))

        # Get statistics
        print("\nAgent Statistics:")
        print(json.dumps(orchestrator.get_agent_statistics(), indent=2))

        print("\nTask Statistics:")
        print(json.dumps(orchestrator.get_task_statistics(), indent=2))

    asyncio.run(main())
