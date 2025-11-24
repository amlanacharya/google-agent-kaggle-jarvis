"""
JARVIS AI Assistant - Google ADK Version
=========================================

This version uses the official Google Agent Development Kit (ADK) for comparison
with our custom POC implementation.

Key Differences from Custom POC:
1. Uses official google-adk library
2. Declarative agent definitions
3. Built-in session management
4. FunctionTool wrappers for tools
5. Runner pattern for execution
6. Evaluation framework included

Author: JARVIS Team
License: MIT
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

# Google ADK imports
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool, google_search
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Optional: Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup from Kaggle secrets.")
except ImportError:
    # Not in Kaggle environment
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    if GOOGLE_API_KEY:
        print(f"✅ Gemini API key configured (ends with: ...{GOOGLE_API_KEY[-8:]})")
    else:
        print("⚠️  Please set GOOGLE_API_KEY environment variable")
except Exception as e:
    print(f"⚠️  Could not load API key from Kaggle secrets: {e}")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configure retry options for robust API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Constants
MEMORY_DB_FILE = "jarvis_memory.json"
USER_ID = "user123"
APP_NAME = "jarvis_adk_assistant"

# Model selection
MODEL_NAME = "gemini-2.0-flash-exp"  # or "gemini-2.5-flash-lite"


# ============================================================================
# MEMORY MANAGEMENT (ADK-style with persistent storage)
# ============================================================================

def load_memory_from_db(user_id: str) -> Dict[str, Any]:
    """Load user memory from persistent storage."""
    try:
        with open(MEMORY_DB_FILE, "r") as f:
            memory_data = json.load(f)
            return memory_data.get(user_id, {
                "interactions": [],
                "preferences": {},
                "context": {}
            })
    except FileNotFoundError:
        return {"interactions": [], "preferences": {}, "context": {}}
    except json.JSONDecodeError:
        print("Error: Invalid JSON in memory database.")
        return {"interactions": [], "preferences": {}, "context": {}}


def save_memory_to_db(user_id: str, memory: Dict[str, Any]):
    """Save user memory to persistent storage."""
    try:
        with open(MEMORY_DB_FILE, "r") as f:
            try:
                memory_data = json.load(f)
            except json.JSONDecodeError:
                memory_data = {}
    except FileNotFoundError:
        memory_data = {}

    memory_data[user_id] = memory
    with open(MEMORY_DB_FILE, "w") as f:
        json.dump(memory_data, f, indent=4)


# ============================================================================
# CUSTOM TOOLS (wrapped with FunctionTool)
# ============================================================================

def store_interaction(
    tool_context: ToolContext,
    user_message: str,
    assistant_response: str,
    metadata: Optional[str] = None
) -> str:
    """Store an interaction in memory."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    memory = load_memory_from_db(user_id)

    interaction = {
        "user": user_message,
        "assistant": assistant_response,
        "timestamp": datetime.now().isoformat(),
        "metadata": json.loads(metadata) if metadata else {}
    }

    memory["interactions"].append(interaction)

    # Keep only last 20 interactions
    if len(memory["interactions"]) > 20:
        memory["interactions"] = memory["interactions"][-20:]

    save_memory_to_db(user_id, memory)
    return f"Stored interaction in memory"


def recall_context(tool_context: ToolContext, query: str, limit: int = 3) -> str:
    """Recall relevant past interactions."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    memory = load_memory_from_db(user_id)

    interactions = memory.get("interactions", [])
    if not interactions:
        return "No previous interactions found."

    # Simple keyword-based search (in production, use embeddings)
    query_lower = query.lower()
    relevant = []

    for interaction in interactions[-limit:]:
        if (query_lower in interaction["user"].lower() or
            query_lower in interaction["assistant"].lower()):
            relevant.append(interaction)

    if not relevant:
        # If no matches, return most recent
        relevant = interactions[-limit:]

    context_parts = []
    for interaction in relevant:
        context_parts.append(
            f"Previous conversation ({interaction['timestamp']}):\n"
            f"User: {interaction['user']}\n"
            f"Assistant: {interaction['assistant']}\n"
        )

    return "\n---\n".join(context_parts)


def analyze_data(tool_context: ToolContext, topic: str, approach: str = "comprehensive") -> str:
    """Perform data analysis on a topic."""
    user_id = tool_context.session_state.get("user_id", USER_ID)

    # This would integrate with actual data analysis tools in production
    # For now, we return a structured prompt for the LLM to handle

    return f"""Performing {approach} analysis on: {topic}

Analysis framework:
1. Define scope and objectives
2. Identify key metrics and KPIs
3. Gather relevant data points
4. Apply analytical methods
5. Draw insights and conclusions
6. Provide actionable recommendations

Please proceed with the analysis based on available context and information."""


def research_topic(tool_context: ToolContext, topic: str, depth: str = "moderate") -> str:
    """Research a topic with specified depth."""
    user_id = tool_context.session_state.get("user_id", USER_ID)

    # Depth levels: quick, moderate, comprehensive
    depth_instructions = {
        "quick": "Provide a brief overview and key points",
        "moderate": "Provide detailed information with context and examples",
        "comprehensive": "Provide in-depth analysis with multiple perspectives and sources"
    }

    instruction = depth_instructions.get(depth, depth_instructions["moderate"])

    return f"""Researching topic: {topic}
Depth level: {depth}
Instructions: {instruction}

Please use available knowledge and search capabilities to research this topic."""


def set_preference(
    tool_context: ToolContext,
    preference_key: str,
    preference_value: str
) -> str:
    """Set a user preference."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    memory = load_memory_from_db(user_id)

    memory["preferences"][preference_key] = preference_value
    save_memory_to_db(user_id, memory)

    return f"Set preference: {preference_key} = {preference_value}"


def get_preferences(tool_context: ToolContext) -> str:
    """Get all user preferences."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    memory = load_memory_from_db(user_id)

    preferences = memory.get("preferences", {})
    if not preferences:
        return "No preferences set."

    pref_list = "\n".join([f"- {k}: {v}" for k, v in preferences.items()])
    return f"User preferences:\n{pref_list}"


# Create FunctionTool wrappers
store_interaction_tool = FunctionTool(store_interaction)
recall_context_tool = FunctionTool(recall_context)
analyze_data_tool = FunctionTool(analyze_data)
research_topic_tool = FunctionTool(research_topic)
set_preference_tool = FunctionTool(set_preference)
get_preferences_tool = FunctionTool(get_preferences)


# ============================================================================
# AGENT DEFINITIONS (ADK-style)
# ============================================================================

# Researcher Agent - specialized in information gathering
researcher_agent = Agent(
    name="ResearcherAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a specialized research agent focused on gathering and synthesizing information.

Your responsibilities:
1. Research topics thoroughly using available search capabilities
2. Provide well-sourced, accurate information
3. Organize findings in a clear, structured manner
4. Cite sources when available
5. Identify knowledge gaps and limitations

Always be thorough but concise. Focus on factual accuracy.""",
    tools=[google_search, research_topic_tool, recall_context_tool],
)

# Analyst Agent - specialized in data analysis
analyst_agent = Agent(
    name="AnalystAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a specialized data analyst agent focused on extracting insights.

Your responsibilities:
1. Analyze data and trends systematically
2. Identify patterns and correlations
3. Provide evidence-based recommendations
4. Use statistical reasoning when appropriate
5. Present findings clearly with visualizations concepts

Always be analytical and objective. Focus on actionable insights.""",
    tools=[analyze_data_tool, recall_context_tool, google_search],
)

# Assistant Agent - general purpose helper
assistant_agent = Agent(
    name="AssistantAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a general-purpose AI assistant inspired by JARVIS from Iron Man.

Your responsibilities:
1. Help users with a wide variety of tasks
2. Maintain context across conversations
3. Be proactive and anticipate user needs
4. Provide clear, helpful responses
5. Learn from user preferences

Be helpful, intelligent, and slightly witty when appropriate.""",
    tools=[
        recall_context_tool,
        set_preference_tool,
        get_preferences_tool,
        google_search,
    ],
)

# Root Agent - orchestrates all specialized agents
root_agent = LlmAgent(
    name="JarvisOrchestrator",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="JARVIS - A sophisticated AI assistant that orchestrates specialized agents",
    instruction="""You are JARVIS, an advanced AI orchestrator that coordinates specialized agents.

Available specialized agents:
- ResearcherAgent: For information gathering and research tasks
- AnalystAgent: For data analysis and insights
- AssistantAgent: For general assistance and conversation

Your responsibilities:
1. Understand user intent and requirements
2. Delegate tasks to the most appropriate specialized agent(s)
3. Synthesize responses from multiple agents when needed
4. Maintain conversation context and memory
5. Provide coherent, helpful responses

When to delegate:
- Research/information queries → ResearcherAgent
- Analysis/insights requests → AnalystAgent
- General help/conversation → AssistantAgent
- Complex tasks → Multiple agents

Always store important interactions in memory for context.""",
    tools=[
        store_interaction_tool,
        recall_context_tool,
        research_topic_tool,
        analyze_data_tool,
        set_preference_tool,
        get_preferences_tool,
        google_search,
    ],
)


# ============================================================================
# SESSION AND RUNNER SETUP
# ============================================================================

session_service = InMemorySessionService()
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def run_session(
    runner_instance: InMemoryRunner,
    user_queries: List[str] | str = None,
    session_name: str = "default",
    verbose: bool = True,
):
    """Run a session with the JARVIS agent."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Session: {session_name}")
        print(f"{'='*80}")

    # Create or fetch session
    try:
        session = await session_service.create_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name
        )
        # Store user_id in session state for tools
        session.state["user_id"] = USER_ID
        await session_service.update_session(session)
    except Exception:
        # If already exists, retrieve it
        session = await session_service.get_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            if verbose:
                print(f"\n🎤 User: {query}")

            query_content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )

            responses = []
            async for event in runner_instance.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=query_content
            ):
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text.strip() and text != "None":
                        responses.append(text)
                        if verbose:
                            print(f"🤖 {event.agent_name or 'JARVIS'}: {text}")

            # Return last response for programmatic use
            return responses[-1] if responses else None
    else:
        if verbose:
            print("No queries provided.")
        return None


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def create_evaluation_set() -> Dict[str, Any]:
    """Create evaluation test cases for JARVIS."""

    eval_set = {
        "eval_set_id": "jarvis_integration_suite",
        "description": "Evaluation cases for JARVIS AI Assistant",
        "eval_cases": [
            {
                "eval_id": "research_task",
                "description": "Test research capabilities",
                "conversation": [
                    {
                        "user_content": {
                            "parts": [{"text": "Research the key differences between transformers and RNNs"}]
                        },
                        "expected_behavior": {
                            "should_use_tools": ["google_search", "research_topic"],
                            "should_mention": ["attention mechanism", "sequential", "parallel"],
                        }
                    }
                ],
            },
            {
                "eval_id": "analysis_task",
                "description": "Test analysis capabilities",
                "conversation": [
                    {
                        "user_content": {
                            "parts": [{"text": "Analyze trends in AI adoption for healthcare"}]
                        },
                        "expected_behavior": {
                            "should_use_tools": ["analyze_data", "google_search"],
                            "should_mention": ["trends", "insights", "healthcare"],
                        }
                    }
                ],
            },
            {
                "eval_id": "memory_task",
                "description": "Test memory and context",
                "conversation": [
                    {
                        "user_content": {
                            "parts": [{"text": "Remember that I prefer technical explanations"}]
                        },
                        "expected_behavior": {
                            "should_use_tools": ["set_preference"],
                        }
                    },
                    {
                        "user_content": {
                            "parts": [{"text": "What are my preferences?"}]
                        },
                        "expected_behavior": {
                            "should_use_tools": ["get_preferences"],
                            "should_mention": ["technical"],
                        }
                    }
                ],
            },
        ],
    }

    # Save evaluation set
    with open("jarvis_adk.evalset.json", "w") as f:
        json.dump(eval_set, f, indent=2)

    print("✅ Created evaluation set: jarvis_adk.evalset.json")
    return eval_set


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

async def run_demo():
    """Run comprehensive demo of JARVIS capabilities."""

    print("\n" + "="*80)
    print(" "*20 + "JARVIS AI ASSISTANT - Google ADK Version")
    print("="*80)
    print("\nInitializing JARVIS system...")

    # Create evaluation set
    create_evaluation_set()

    print("\n✅ JARVIS initialized successfully!")

    # Demo scenarios
    demos = [
        {
            "name": "Research & Information Gathering",
            "query": "What are the key differences between transformer and RNN architectures in deep learning?",
            "session": "demo_research"
        },
        {
            "name": "Data Analysis & Insights",
            "query": "Analyze the trend of AI adoption in healthcare from 2020-2024",
            "session": "demo_analysis"
        },
        {
            "name": "Preference Learning",
            "query": "Remember that I prefer concise, technical explanations with code examples",
            "session": "demo_memory"
        },
        {
            "name": "Context Awareness",
            "query": "Based on my preferences, explain what vector embeddings are",
            "session": "demo_memory"
        },
        {
            "name": "Multi-capability Task",
            "query": "Research quantum computing for AI, analyze its potential impact, and store your findings",
            "session": "demo_multi"
        }
    ]

    for i, demo in enumerate(demos, 1):
        print("\n" + "="*80)
        print(f"DEMO {i}: {demo['name']}")
        print("="*80)

        await run_session(
            runner,
            user_queries=demo['query'],
            session_name=demo['session'],
            verbose=True
        )

        # Small delay between demos
        await asyncio.sleep(1)

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - All capabilities demonstrated!")
    print("="*80)
    print("\nDemonstrated Capabilities:")
    print("  1. ✓ Research with Google Search integration")
    print("  2. ✓ Data analysis and insights")
    print("  3. ✓ Memory and preference learning")
    print("  4. ✓ Context-aware responses")
    print("  5. ✓ Multi-agent orchestration")
    print("  6. ✓ Tool usage and delegation")
    print()


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

async def interactive_mode():
    """Run JARVIS in interactive mode."""

    print("\n" + "="*80)
    print(" "*20 + "JARVIS INTERACTIVE MODE (ADK)")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question or task")
    print("  - Type 'prefs' to see your preferences")
    print("  - Type 'memory' to see conversation history")
    print("  - Type 'quit' to exit")
    print()

    session_name = "interactive_session"

    # Ensure session exists
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_name
        )
    except Exception:
        pass

    while True:
        try:
            user_input = input("\n🎤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'prefs':
                memory = load_memory_from_db(USER_ID)
                prefs = memory.get("preferences", {})
                print(f"\n📊 Preferences: {json.dumps(prefs, indent=2)}")
                continue

            if user_input.lower() == 'memory':
                memory = load_memory_from_db(USER_ID)
                interactions = memory.get("interactions", [])
                print(f"\n🧠 Memory: {len(interactions)} interactions stored")
                continue

            # Process with JARVIS
            print("\n🤔 Processing...")
            await run_session(
                runner,
                user_queries=user_input,
                session_name=session_name,
                verbose=True
            )

        except EOFError:
            print("\n⚠️  Interactive mode is not supported in Jupyter/Kaggle notebooks.")
            print("Please run the automated demo instead (option 1).")
            break
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""

    # Ensure session service is ready
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id="init_session"
        )
        print("✅ Session service initialized")
    except Exception:
        pass

    # Choose mode
    print("\n" + "="*80)
    print("JARVIS AI ASSISTANT - Google ADK Version")
    print("="*80)
    print("\nSelect mode:")
    print("  1. Run automated demo (recommended)")
    print("  2. Interactive mode")
    print()

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            await run_demo()
        elif choice == "2":
            await interactive_mode()
        else:
            print("Running automated demo (default)...")
            await run_demo()

    except EOFError:
        # Running in Jupyter/Kaggle notebook - no interactive input available
        print("📓 Running in notebook environment - starting automated demo...")
        print()
        await run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# SAFE EXECUTION (handles both script and notebook environments)
# ============================================================================

def run_main_safely():
    """Run main safely in both script and interactive environments."""
    try:
        # If no running loop, this raises RuntimeError
        loop = asyncio.get_running_loop()
        # We're in a notebook with a running loop
        # Try to apply nest_asyncio to allow nested loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
            print("📓 Notebook environment detected, applying nest_asyncio...")
            asyncio.run(main())
        except ImportError:
            # nest_asyncio not available, schedule as task
            print("🔄 Notebook environment detected (install nest_asyncio for better support)")
            print("📌 Scheduling task... Please wait for execution to complete.")
            task = asyncio.ensure_future(main())
            return task
    except RuntimeError:
        # Normal script execution
        asyncio.run(main())


if __name__ == "__main__":
    _run_task = run_main_safely()
    # Note: In notebooks without nest_asyncio, you may need to await _run_task
