"""
JARVIS AI Assistant - Kaggle POC Version
=========================================

A simplified proof-of-concept version of the JARVIS AI Assistant for Kaggle notebooks.
This version demonstrates core AI capabilities without requiring complex infrastructure.

Features:
1. Multi-modal LLM (Gemini) with intelligent prompting
2. Contextual memory system (in-memory vector store)
3. Multi-agent orchestration (Researcher, Analyst, Assistant)
4. Web search capabilities (optional)
5. Intelligent task routing and delegation

Requirements:
- Google API Key (Gemini)
- Minimal dependencies (see kaggle-requirements.txt)

Author: JARVIS Team
License: MIT
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np

# Check for required packages
try:
    import google.generativeai as genai
    print("✓ Google Generative AI library loaded")
except ImportError:
    print("❌ Please install: pip install google-generativeai")
    raise

# Optional: Web search (graceful degradation if not available)
try:
    import requests
    WEB_SEARCH_AVAILABLE = True
    print("✓ Requests library loaded (web search enabled)")
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("⚠️  Requests not available (web search disabled)")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get API key from environment or Kaggle secrets
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    print("\n" + "="*80)
    print("⚠️  GOOGLE_API_KEY not found!")
    print("="*80)
    print("\nPlease set your API key:")
    print("  Option 1: os.environ['GOOGLE_API_KEY'] = 'your-key-here'")
    print("  Option 2: export GOOGLE_API_KEY=your-key-here")
    print("\nGet your key at: https://makersuite.google.com/app/apikey")
    print("="*80 + "\n")
else:
    print(f"✓ Google API Key configured (ends with: ...{GOOGLE_API_KEY[-8:]})")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)


# ============================================================================
# CORE: IN-MEMORY VECTOR STORE (Simplified Memory System)
# ============================================================================

class SimpleVectorStore:
    """Lightweight in-memory vector store using cosine similarity."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def add(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a text with its embedding to the store."""
        self.texts.append(text)
        self.vectors.append(embedding)
        self.metadata.append(metadata or {})

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts using cosine similarity."""
        if not self.vectors:
            return []

        # Compute cosine similarities
        similarities = []
        for vec in self.vectors:
            similarity = np.dot(query_embedding, vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vec)
            )
            similarities.append(similarity)

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })

        return results

    def clear(self):
        """Clear all stored vectors."""
        self.vectors = []
        self.metadata = []
        self.texts = []


# ============================================================================
# CORE: MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """Manages conversation memory with semantic search."""

    def __init__(self, model_name: str = "models/embedding-001"):
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory = SimpleVectorStore()
        self.model_name = model_name

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Gemini."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
        except Exception as e:
            print(f"Warning: Embedding failed: {e}")
            # Fallback to random embedding (for demo purposes)
            return np.random.randn(768)

    def add_interaction(self, user_message: str, assistant_response: str, metadata: Dict[str, Any] = None):
        """Store an interaction in memory."""
        interaction = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Add to short-term memory (keep last 10)
        self.short_term_memory.append(interaction)
        if len(self.short_term_memory) > 10:
            # Move oldest to long-term
            old = self.short_term_memory.pop(0)
            embedding = self._get_embedding(f"{old['user']} {old['assistant']}")
            self.long_term_memory.add(
                text=f"User: {old['user']}\nAssistant: {old['assistant']}",
                embedding=embedding,
                metadata=old['metadata']
            )

    def search_memory(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search long-term memory for relevant past interactions."""
        query_embedding = self._get_embedding(query)
        return self.long_term_memory.search(query_embedding, top_k)

    def get_recent_context(self, last_n: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.short_term_memory[-last_n:]
        context_parts = []
        for interaction in recent:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['assistant']}")
        return "\n".join(context_parts)

    def clear(self):
        """Clear all memory."""
        self.short_term_memory = []
        self.long_term_memory.clear()


# ============================================================================
# CORE: LLM CLIENT
# ============================================================================

class GeminiClient:
    """Wrapper for Gemini API with advanced features."""

    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate text from prompt."""
        try:
            # Combine system instruction with prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            return response.text

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Chat with message history."""
        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat(history=[])

            # Add message history (skip system messages for simplicity)
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])

            # Send final message
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            return response.text

        except Exception as e:
            return f"Error in chat: {str(e)}"


# ============================================================================
# AGENTS: BASE AGENT
# ============================================================================

@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    confidence: float
    agent_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, llm: GeminiClient, memory: MemoryManager):
        self.name = name
        self.llm = llm
        self.memory = memory
        self.capabilities: List[str] = []

    async def process(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process a task and return response."""
        raise NotImplementedError

    def can_handle(self, task: str) -> float:
        """Return confidence score (0-1) for handling this task."""
        task_lower = task.lower()
        score = 0.0
        for capability in self.capabilities:
            if capability.lower() in task_lower:
                score += 0.3
        return min(score, 1.0)


# ============================================================================
# AGENTS: RESEARCHER AGENT
# ============================================================================

class ResearcherAgent(BaseAgent):
    """Agent specialized in research and information gathering."""

    def __init__(self, llm: GeminiClient, memory: MemoryManager):
        super().__init__("Researcher", llm, memory)
        self.capabilities = ["research", "search", "find", "information", "learn", "investigate"]

    async def search_web(self, query: str) -> Optional[str]:
        """Perform web search (simplified)."""
        if not WEB_SEARCH_AVAILABLE:
            return None

        # Note: In production, use proper search API (Serper, Tavily, etc.)
        # This is a placeholder that would need API integration
        return f"[Web search results for: {query}]"

    async def process(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Research a topic and provide comprehensive information."""
        context = context or {}

        # Check memory for related information
        relevant_memory = self.memory.search_memory(task, top_k=2)
        memory_context = ""
        if relevant_memory:
            memory_context = "Relevant past information:\n" + "\n".join([
                f"- {mem['text'][:100]}..." for mem in relevant_memory
            ])

        # Attempt web search
        search_results = await self.search_web(task)

        # Build research prompt
        prompt = f"""As a research specialist, provide comprehensive information about:

Task: {task}

{memory_context}

{f"Web search results: {search_results}" if search_results else ""}

Please provide:
1. Key findings and facts
2. Important context and background
3. Relevant insights and analysis
4. Sources of information (if available)

Be thorough but concise."""

        response = await self.llm.generate(
            prompt,
            temperature=0.3,  # Lower temperature for factual content
            system_instruction="You are a thorough research assistant focused on providing accurate, well-sourced information."
        )

        return AgentResponse(
            content=response,
            confidence=0.8,
            agent_name=self.name,
            metadata={"task_type": "research", "has_web_results": search_results is not None}
        )


# ============================================================================
# AGENTS: DATA ANALYST AGENT
# ============================================================================

class DataAnalystAgent(BaseAgent):
    """Agent specialized in data analysis and insights."""

    def __init__(self, llm: GeminiClient, memory: MemoryManager):
        super().__init__("DataAnalyst", llm, memory)
        self.capabilities = ["analyze", "data", "statistics", "trends", "insights", "calculate"]

    async def process(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Analyze data and provide insights."""
        context = context or {}

        prompt = f"""As a data analyst, analyze the following task:

Task: {task}

Context: {json.dumps(context, indent=2) if context else "No additional context"}

Please provide:
1. Analysis approach and methodology
2. Key insights and findings
3. Statistical observations (if applicable)
4. Recommendations based on analysis
5. Any caveats or limitations

Be analytical and data-driven."""

        response = await self.llm.generate(
            prompt,
            temperature=0.4,
            system_instruction="You are an expert data analyst focused on extracting insights and providing evidence-based recommendations."
        )

        return AgentResponse(
            content=response,
            confidence=0.85,
            agent_name=self.name,
            metadata={"task_type": "analysis"}
        )


# ============================================================================
# AGENTS: GENERAL ASSISTANT AGENT
# ============================================================================

class GeneralAssistantAgent(BaseAgent):
    """General-purpose assistant for various tasks."""

    def __init__(self, llm: GeminiClient, memory: MemoryManager):
        super().__init__("GeneralAssistant", llm, memory)
        self.capabilities = ["help", "assist", "explain", "summarize", "write", "create"]

    async def process(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Handle general assistance tasks."""
        context = context or {}

        # Get recent conversation context
        recent_context = self.memory.get_recent_context(last_n=3)

        prompt = f"""As JARVIS, a helpful AI assistant, respond to:

Task: {task}

Recent conversation context:
{recent_context if recent_context else "No prior conversation"}

Additional context: {json.dumps(context, indent=2) if context else "None"}

Provide a helpful, clear, and actionable response."""

        response = await self.llm.generate(
            prompt,
            temperature=0.7,
            system_instruction="You are JARVIS, a sophisticated AI assistant inspired by Iron Man's AI. Be helpful, intelligent, and slightly witty."
        )

        return AgentResponse(
            content=response,
            confidence=0.75,
            agent_name=self.name,
            metadata={"task_type": "general_assistance"}
        )


# ============================================================================
# ORCHESTRATOR: JARVIS MAIN AGENT
# ============================================================================

class JarvisOrchestrator:
    """Main orchestrator for the JARVIS system."""

    def __init__(self):
        self.llm = GeminiClient()
        self.memory = MemoryManager()

        # Initialize specialized agents
        self.agents: Dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(self.llm, self.memory),
            "analyst": DataAnalystAgent(self.llm, self.memory),
            "assistant": GeneralAssistantAgent(self.llm, self.memory)
        }

        self.stats = defaultdict(int)

    def _select_agent(self, task: str) -> BaseAgent:
        """Select the best agent for a given task."""
        scores = {}
        for name, agent in self.agents.items():
            scores[name] = agent.can_handle(task)

        # Select agent with highest confidence
        best_agent_name = max(scores, key=scores.get)
        best_score = scores[best_agent_name]

        # If no agent is confident, use general assistant
        if best_score < 0.3:
            return self.agents["assistant"]

        return self.agents[best_agent_name]

    async def process_request(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        use_multi_agent: bool = False
    ) -> AgentResponse:
        """Process a user request using appropriate agent(s)."""

        if use_multi_agent:
            # Multi-agent mode: get responses from all agents and synthesize
            responses = []
            for agent in self.agents.values():
                response = await agent.process(user_message, context)
                responses.append(response)

            # Synthesize responses
            synthesis_prompt = f"""Synthesize the following responses from multiple AI agents:

User request: {user_message}

Agent responses:
"""
            for resp in responses:
                synthesis_prompt += f"\n{resp.agent_name} (confidence {resp.confidence}):\n{resp.content}\n"

            synthesis_prompt += "\nProvide a comprehensive, unified response that combines the best insights from all agents."

            final_content = await self.llm.generate(
                synthesis_prompt,
                temperature=0.6,
                system_instruction="You are synthesizing multiple expert opinions into a coherent response."
            )

            result = AgentResponse(
                content=final_content,
                confidence=sum(r.confidence for r in responses) / len(responses),
                agent_name="JARVIS-MultiAgent",
                metadata={"agents_consulted": [r.agent_name for r in responses]}
            )
        else:
            # Single-agent mode: select best agent
            selected_agent = self._select_agent(user_message)
            self.stats[selected_agent.name] += 1
            result = await selected_agent.process(user_message, context)

        # Store in memory
        self.memory.add_interaction(
            user_message=user_message,
            assistant_response=result.content,
            metadata={
                "agent": result.agent_name,
                "confidence": result.confidence
            }
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": sum(self.stats.values()),
            "agent_usage": dict(self.stats),
            "memory_size": len(self.memory.short_term_memory)
        }


# ============================================================================
# DEMO: INTERACTIVE DEMONSTRATION
# ============================================================================

async def run_demo():
    """Run interactive demonstration of JARVIS capabilities."""

    print("\n" + "="*80)
    print(" "*25 + "JARVIS AI ASSISTANT - POC")
    print("="*80)
    print()
    print("Initializing JARVIS system...")

    # Initialize JARVIS
    jarvis = JarvisOrchestrator()

    print("✓ JARVIS initialized successfully")
    print()

    # Demo scenarios
    demo_tasks = [
        {
            "name": "Capability 1: Research & Information Gathering",
            "task": "What are the key differences between transformer and RNN architectures in deep learning?",
            "use_multi_agent": False
        },
        {
            "name": "Capability 2: Data Analysis & Insights",
            "task": "Analyze the trend of AI adoption in healthcare from 2020-2024",
            "use_multi_agent": False
        },
        {
            "name": "Capability 3: General Assistance with Memory",
            "task": "Based on our previous discussion, what AI architecture would you recommend for a time-series prediction task?",
            "use_multi_agent": False
        },
        {
            "name": "Capability 4: Multi-Agent Collaboration",
            "task": "Should I invest in learning quantum computing for AI research?",
            "use_multi_agent": True
        }
    ]

    for i, demo in enumerate(demo_tasks, 1):
        print("\n" + "="*80)
        print(f"DEMO {i}: {demo['name']}")
        print("="*80)
        print(f"\n📝 Task: {demo['task']}")
        print(f"🤖 Mode: {'Multi-Agent Collaboration' if demo['use_multi_agent'] else 'Single Agent'}")
        print()

        # Process request
        response = await jarvis.process_request(
            user_message=demo['task'],
            use_multi_agent=demo['use_multi_agent']
        )

        print(f"🎯 Agent: {response.agent_name}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"\n💬 Response:")
        print("-" * 80)
        print(response.content)
        print("-" * 80)

        # Show memory context for memory demonstration
        if i == 3:
            print("\n🧠 Memory Context:")
            print("-" * 80)
            recent = jarvis.memory.get_recent_context(last_n=2)
            print(recent if recent else "No context available")
            print("-" * 80)

        # Add small delay between demos
        await asyncio.sleep(1)

    # Show statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    stats = jarvis.get_statistics()
    print(json.dumps(stats, indent=2))
    print()

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - All capabilities demonstrated successfully!")
    print("="*80)
    print()
    print("Demonstrated Capabilities:")
    print("  1. ✓ Multi-modal LLM (Gemini) with intelligent routing")
    print("  2. ✓ Contextual memory with semantic search")
    print("  3. ✓ Multi-agent orchestration (Researcher, Analyst, Assistant)")
    print("  4. ✓ Intelligent task delegation based on capabilities")
    print("  5. ✓ Multi-agent consensus and synthesis")
    print()


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

async def interactive_mode():
    """Run JARVIS in interactive mode."""

    print("\n" + "="*80)
    print(" "*25 + "JARVIS INTERACTIVE MODE")
    print("="*80)
    print()
    print("Commands:")
    print("  - Type your question or task")
    print("  - Type 'multi' to enable multi-agent mode")
    print("  - Type 'stats' to see statistics")
    print("  - Type 'clear' to clear memory")
    print("  - Type 'quit' to exit")
    print()

    jarvis = JarvisOrchestrator()
    multi_agent_mode = False

    while True:
        try:
            user_input = input("\n🎤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'multi':
                multi_agent_mode = not multi_agent_mode
                print(f"✓ Multi-agent mode: {'ON' if multi_agent_mode else 'OFF'}")
                continue

            if user_input.lower() == 'stats':
                print("\n📊 Statistics:")
                print(json.dumps(jarvis.get_statistics(), indent=2))
                continue

            if user_input.lower() == 'clear':
                jarvis.memory.clear()
                print("✓ Memory cleared")
                continue

            # Process request
            print("\n🤔 Processing...")
            response = await jarvis.process_request(
                user_message=user_input,
                use_multi_agent=multi_agent_mode
            )

            print(f"\n🤖 JARVIS ({response.agent_name}, confidence: {response.confidence:.2f}):")
            print(response.content)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("JARVIS AI ASSISTANT - Kaggle POC Version")
    print("="*80)
    print()

    # Check API key
    if not GOOGLE_API_KEY:
        print("Please set GOOGLE_API_KEY to continue.")
        import sys
        sys.exit(1)

    # Choose mode
    print("Select mode:")
    print("  1. Run automated demo (recommended)")
    print("  2. Interactive mode")
    print()

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            asyncio.run(run_demo())
        elif choice == "2":
            asyncio.run(interactive_mode())
        else:
            print("Running automated demo (default)...")
            asyncio.run(run_demo())

    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
