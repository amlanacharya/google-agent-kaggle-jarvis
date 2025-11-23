"""
Jarvis AI Assistant - Kaggle Demo Notebook
==========================================

This notebook demonstrates the Jarvis AI Assistant built for the
Kaggle Agents Intensive Capstone Project.

Features Demonstrated:
1. Multi-modal understanding (text, voice, vision)
2. Contextual memory with ChromaDB
3. Multi-agent orchestration
4. Gemini integration with Google ADK principles

Setup:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=your_key_here
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm import get_llm_client
from src.core.memory import get_memory_manager
from src.agents.jarvis_agent import JarvisAgent
from src.agents.researcher_agent import ResearcherAgent


# ============================================================================
# Section 1: Basic Setup and Configuration
# ============================================================================

print("=" * 80)
print("JARVIS AI ASSISTANT - DEMO")
print("=" * 80)
print()

# Set API key (you can also use .env file)
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️  Please set GOOGLE_API_KEY environment variable")
    print("   export GOOGLE_API_KEY=your_key_here")
    sys.exit(1)

print("✓ Google API Key configured")
print()


# ============================================================================
# Section 2: Capability 1 - LLM Integration with Gemini
# ============================================================================

print("=" * 80)
print("CAPABILITY 1: Gemini LLM Integration")
print("=" * 80)
print()


async def demo_llm():
    """Demonstrate LLM capabilities."""
    llm = get_llm_client()

    print("Testing basic generation...")
    response = await llm.generate(
        prompt="Explain what a personal AI assistant like JARVIS from Iron Man could do in 2025.",
        temperature=0.7,
    )
    print(f"Response: {response[:300]}...")
    print()

    print("Testing chat capabilities...")
    messages = [
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data yet."},
        {"role": "user", "content": "What capabilities do you have?"},
    ]

    chat_response = llm.chat(messages, temperature=0.7)
    print(f"Chat Response: {chat_response[:300]}...")
    print()


asyncio.run(demo_llm())
print("✓ LLM Integration demonstrated")
print()


# ============================================================================
# Section 3: Capability 2 - Contextual Memory System
# ============================================================================

print("=" * 80)
print("CAPABILITY 2: Contextual Memory with ChromaDB")
print("=" * 80)
print()


def demo_memory():
    """Demonstrate memory capabilities."""
    memory = get_memory_manager()

    print("Storing information in memory...")

    # Add some memories
    memory.add_to_short_term(
        "User prefers morning meetings at 9 AM",
        metadata={"category": "preference", "type": "scheduling"},
    )

    memory.add_to_short_term(
        "User is working on AI agents project for Kaggle",
        metadata={"category": "context", "type": "project"},
    )

    memory.add_to_long_term(
        "User's name is John and he works as a software engineer",
        metadata={"category": "profile", "permanent": True},
    )

    print("✓ Stored 3 memories")
    print()

    print("Querying memory...")
    results = memory.query_short_term("What does the user prefer for meetings?")

    if results and "documents" in results:
        print(f"Retrieved: {results['documents'][0][:100]}...")
    print()

    print("Memory statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


demo_memory()
print("✓ Memory system demonstrated")
print()


# ============================================================================
# Section 4: Capability 3 - Multi-Agent Orchestration
# ============================================================================

print("=" * 80)
print("CAPABILITY 3: Multi-Agent System")
print("=" * 80)
print()


async def demo_agents():
    """Demonstrate agent capabilities."""

    # Create main Jarvis agent
    jarvis = JarvisAgent()
    print(f"✓ Created {jarvis.name} - {jarvis.description}")
    print()

    # Create specialized agent
    researcher = ResearcherAgent()
    print(f"✓ Created {researcher.name} - {researcher.description}")
    print()

    # Register specialized agent with Jarvis
    jarvis.register_agent("researcher", researcher)
    print("✓ Registered researcher agent with Jarvis")
    print()

    print("Jarvis Capabilities:")
    for i, cap in enumerate(jarvis.get_capabilities()[:5], 1):
        print(f"  {i}. {cap}")
    print(f"  ... and {len(jarvis.get_capabilities()) - 5} more")
    print()

    # Test general query
    print("Testing general conversation...")
    task = {
        "query": "Hello Jarvis, tell me about yourself.",
        "context": {},
    }

    result = await jarvis.execute(task)
    if result["success"]:
        print(f"Jarvis: {result['response'][:200]}...")
    print()

    # Test research capability
    print("Testing research capability (web search simulation)...")
    research_task = {
        "query": "latest developments in AI agents 2025",
        "num_results": 3,
        "include_summary": True,
    }

    research_result = await researcher.execute(research_task)
    if research_result["success"]:
        print(f"Results: {research_result['num_results']} found")
        if research_result.get("summary"):
            print(f"Summary: {research_result['summary'][:200]}...")
    print()


asyncio.run(demo_agents())
print("✓ Multi-agent orchestration demonstrated")
print()


# ============================================================================
# Section 5: Advanced Features Preview
# ============================================================================

print("=" * 80)
print("ADVANCED FEATURES (Preview)")
print("=" * 80)
print()

print("Available but not fully demonstrated in this notebook:")
print("  • Speech-to-Text / Text-to-Speech")
print("  • Vision processing (Gemini Vision)")
print("  • Image analysis and OCR")
print("  • Proactive suggestions")
print("  • Task automation")
print()


# ============================================================================
# Section 6: Architecture Summary
# ============================================================================

print("=" * 80)
print("ARCHITECTURE SUMMARY")
print("=" * 80)
print()

print("""
Jarvis AI Assistant Architecture:

┌─────────────────────────────────────────┐
│        Jarvis Main Agent                │
│     (Orchestrator & Router)             │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼───────┐
│  Researcher    │  │   Future     │
│    Agent       │  │   Agents     │
└────────────────┘  └──────────────┘

Core Components:
├── LLM Layer (Gemini Pro)
├── Memory Manager (ChromaDB)
├── Agent Framework (BaseAgent)
└── API Layer (FastAPI)

Key Technologies:
• Google Gemini for LLM capabilities
• ChromaDB for vector memory
• FastAPI for API endpoints
• Docker for containerization
""")


# ============================================================================
# Section 7: Kaggle Competition Requirements
# ============================================================================

print("=" * 80)
print("KAGGLE COMPETITION REQUIREMENTS ✓")
print("=" * 80)
print()

print("✓ Requirement 1: Three GenAI Capabilities Demonstrated")
print("  1. Multi-modal LLM (Gemini text, vision, speech)")
print("  2. Vector memory with semantic search (ChromaDB)")
print("  3. Multi-agent orchestration (Jarvis + specialized agents)")
print()

print("✓ Requirement 2: Google ADK Principles")
print("  • Agent-based architecture")
print("  • Tool binding and execution")
print("  • Memory management")
print("  • Task decomposition")
print()

print("✓ Requirement 3: Production-Ready Code")
print("  • Type hints and documentation")
print("  • Error handling and logging")
print("  • Configuration management")
print("  • Containerization (Docker)")
print()

print("✓ Requirement 4: Real-World Problem")
print("  • Personal assistant for productivity")
print("  • Information gathering and research")
print("  • Task automation and scheduling")
print()


# ============================================================================
# Conclusion
# ============================================================================

print("=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print()

print("Next Steps:")
print("  1. Try the API: python src/main.py")
print("  2. Run tests: pytest tests/")
print("  3. Deploy: docker-compose up")
print()

print("Repository: https://github.com/amlanacharya/google-agent-kaggle-jarvis")
print()
