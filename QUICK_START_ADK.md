# 🚀 JARVIS Google ADK Version - Quick Start

## Overview

This is the **Google ADK-based** implementation of JARVIS, using the official Google Agent Development Kit.

**File:** `notebooks/jarvis_adk_version.py`

---

## ⚡ Quick Comparison

| Feature | Custom POC | ADK Version (This) |
|---------|-----------|-------------------|
| Dependencies | 2 packages (~50MB) | 3+ packages (~200MB) |
| Setup Time | <1 min | 2-3 min |
| Code Complexity | Simple | Framework patterns |
| Official Support | ❌ | ✅ Google-backed |
| Best for | Learning, Kaggle | Production apps |

---

## 📦 Installation

### Option 1: Kaggle Notebook

```python
# In Kaggle notebook
!pip install -q google-adk google-generativeai numpy

import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ['GOOGLE_API_KEY'] = user_secrets.get_secret("GOOGLE_API_KEY")
```

### Option 2: Local Environment

```bash
# Install dependencies
pip install -r adk-requirements.txt

# Set API key
export GOOGLE_API_KEY='your-google-api-key'

# Run
python notebooks/jarvis_adk_version.py
```

---

## 🎯 Key Differences from Custom POC

### 1. Agent Definition

**Custom POC:**
```python
class ResearcherAgent(BaseAgent):
    def __init__(self, llm, memory):
        super().__init__("Researcher", llm, memory)

    async def process(self, task, context):
        # Custom logic
        pass
```

**Google ADK:**
```python
researcher_agent = Agent(
    name="ResearcherAgent",
    model=Gemini(model="gemini-2.0-flash-exp"),
    instruction="""You are a research specialist...""",
    tools=[google_search, research_topic_tool]
)
```

### 2. Tools

**Custom POC:**
```python
# Direct implementation
async def search_web(self, query: str):
    return results
```

**Google ADK:**
```python
# Wrapped with FunctionTool
def research_topic(tool_context: ToolContext, topic: str):
    user_id = tool_context.session_state.get("user_id")
    return results

research_topic_tool = FunctionTool(research_topic)
```

### 3. Execution

**Custom POC:**
```python
jarvis = JarvisOrchestrator()
response = await jarvis.process_request("Hello")
print(response.content)
```

**Google ADK:**
```python
runner = InMemoryRunner(agent=root_agent, app_name="jarvis")

async for event in runner.run_async(
    user_id=USER_ID,
    session_id=session.id,
    new_message=query_content
):
    print(event.content)
```

---

## 🏗️ Architecture

```
┌────────────────────────────┐
│   LlmAgent (Root)          │
│   - JarvisOrchestrator     │
│   - Automatic delegation   │
└────────┬───────────────────┘
         │
    ┌────▼────────┐
    │ InMemory    │
    │  Runner     │
    └────┬────────┘
         │
    ┌────┴────────────────────┐
    │                         │
┌───▼────────┐  ┌────────┐  ┌──────────┐
│Researcher  │  │Analyst │  │Assistant │
│  Agent     │  │ Agent  │  │  Agent   │
└───┬────────┘  └───┬────┘  └────┬─────┘
    │               │            │
    └───────────────┴────────────┘
                    │
              ┌─────▼─────┐
              │  Gemini   │
              │(ADK Model)│
              └─────┬─────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────▼──────┐      ┌──────▼──────┐
    │InMemory   │      │  Session    │
    │Session    │      │   State     │
    │ Service   │      │  (Tools)    │
    └───────────┘      └─────────────┘
```

---

## 🎮 Usage Examples

### Run Demo

```bash
python notebooks/jarvis_adk_version.py
# Select option 1 for demo
```

**Demo Output:**
```
================================================================================
DEMO 1: Research & Information Gathering
================================================================================

🎤 User: What are the key differences between transformer and RNN...

🤖 ResearcherAgent: [Detailed research response using google_search]
```

### Interactive Mode

```bash
python notebooks/jarvis_adk_version.py
# Select option 2 for interactive

🎤 You: Research quantum computing
🤔 Processing...
🤖 JARVIS: [Response from appropriate agent]
```

### Programmatic Use

```python
from jarvis_adk_version import runner, session_service, run_session

# Run a query
response = await run_session(
    runner,
    user_queries="Analyze AI trends",
    session_name="my_session",
    verbose=True
)
```

---

## 🛠️ Key Features

### 1. Official Google Tools

```python
# Built-in Google Search
tools=[google_search]  # No manual implementation needed!
```

### 2. Session Management

```python
# Automatic session handling
session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id="my_session"
)

# Access in tools
def my_tool(tool_context: ToolContext):
    user_id = tool_context.session_state.get("user_id")
```

### 3. Retry Configuration

```python
# Built-in retry logic
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

model = Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config)
```

### 4. Evaluation Framework

```python
# Create test cases
eval_set = {
    "eval_set_id": "jarvis_suite",
    "eval_cases": [
        {
            "eval_id": "research_task",
            "conversation": [{
                "user_content": {"parts": [{"text": "Research topic X"}]},
                "expected_behavior": {
                    "should_use_tools": ["google_search"],
                    "should_mention": ["key", "concepts"]
                }
            }]
        }
    ]
}
```

---

## 📊 Demonstrated Capabilities

✅ **6+ GenAI Capabilities:**
1. **Multi-modal LLM** (Gemini Pro)
2. **Tool calling** (FunctionTool framework)
3. **Multi-agent orchestration** (LlmAgent delegation)
4. **Session management** (InMemorySessionService)
5. **Memory persistence** (JSON database)
6. **Web search integration** (Built-in google_search)
7. **Evaluation framework** (Test cases)

---

## 🔧 Customization

### Add Your Own Agent

```python
my_agent = Agent(
    name="MyCustomAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""Your instructions here...""",
    tools=[your_custom_tools]
)

# Add to root agent's tools
root_agent = LlmAgent(
    tools=[...existing_tools..., my_agent_tool]
)
```

### Add Custom Tool

```python
def my_custom_tool(tool_context: ToolContext, param: str) -> str:
    """Your tool logic."""
    user_id = tool_context.session_state.get("user_id")
    # Your implementation
    return "Result"

# Wrap and register
my_tool = FunctionTool(my_custom_tool)

agent = Agent(
    name="MyAgent",
    tools=[my_tool, google_search]
)
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Cold start | ~3s (framework init) |
| Response time | 2-5s (same as custom) |
| Memory usage | ~150MB (framework) |
| Install time | 2-3 minutes |

---

## 🎯 When to Use ADK Version

✅ **Use this version when:**
- Building production applications
- Need official Google support
- Want built-in best practices
- Working in teams (standard patterns)
- Need formal evaluation framework
- Want built-in tools (google_search, etc.)

❌ **Don't use if:**
- Just learning (custom POC is simpler)
- Resource constrained (Kaggle free tier)
- Need minimal dependencies
- Want full control over logic
- Quick prototype/demo

---

## 🐛 Troubleshooting

### Error: "google-adk not found"

```bash
pip install google-adk
```

### Error: "Session already exists"

```python
# Normal - ADK reuses sessions
# Just continue, or delete the session:
await session_service.delete_session(app_name, user_id, session_id)
```

### Slow installation

**Expected!** ADK has many dependencies (~200MB). Consider:
- Use `pip install -q` to reduce output
- Install once, reuse environment
- For Kaggle: Add to notebook setup cell

---

## 📚 Resources

- **Google ADK Docs:** (Check Google AI documentation)
- **Gemini API:** https://ai.google.dev/docs
- **Comparison:** See `ADK_COMPARISON.md`
- **Custom POC:** See `notebooks/jarvis_kaggle_poc.py`

---

## 🔄 Migration from Custom POC

If you built with Custom POC and want ADK features:

**1. Keep your tools** - Wrap with `FunctionTool`:
```python
# Your existing function
def my_existing_tool(param: str) -> str:
    return result

# Wrap it
my_tool = FunctionTool(my_existing_tool)
```

**2. Convert agents** - From classes to declarations:
```python
# Before (Custom)
class MyAgent(BaseAgent):
    ...

# After (ADK)
my_agent = Agent(
    name="MyAgent",
    instruction="...",
    tools=[...]
)
```

**3. Use sessions** - Replace custom memory:
```python
# Access session state in tools
def my_tool(tool_context: ToolContext):
    state = tool_context.session_state
```

**Time:** ~4 hours for JARVIS-sized project

---

## ✅ Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r adk-requirements.txt
   ```

2. **Set API key:**
   ```bash
   export GOOGLE_API_KEY='your-key'
   ```

3. **Run demo:**
   ```bash
   python notebooks/jarvis_adk_version.py
   ```

4. **Compare with custom POC:**
   - Read `ADK_COMPARISON.md`
   - Try both versions
   - Choose based on your needs!

---

**Built for Kaggle Agents Intensive Capstone Project**

*Two implementations, both production-ready!* 🚀

- **Custom POC:** Simple, educational, flexible
- **Google ADK:** Official, production, best practices

**Choose wisely!** 💡
