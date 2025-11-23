# JARVIS AI Assistant - Kaggle POC Version 🤖

A lightweight, production-ready proof-of-concept of the JARVIS AI Assistant system, designed specifically for Kaggle notebooks with minimal dependencies.

## 🎯 Overview

This POC demonstrates a sophisticated multi-agent AI system inspired by Iron Man's JARVIS, showcasing:

- **Multi-modal LLM Integration** (Gemini)
- **Contextual Memory System** with semantic search
- **Multi-Agent Orchestration** with intelligent task routing
- **Real-world AI Capabilities** without complex infrastructure

## ✨ Key Features

### 1. **Intelligent Agent System**
- **Researcher Agent**: Specializes in information gathering and research
- **Data Analyst Agent**: Focuses on analysis and insights
- **General Assistant Agent**: Handles diverse tasks with conversation context

### 2. **Memory Management**
- Short-term memory for recent conversations (last 10 interactions)
- Long-term memory with semantic vector search
- Automatic memory consolidation and retrieval

### 3. **Smart Task Routing**
- Capability-based agent selection
- Confidence scoring for optimal task delegation
- Multi-agent collaboration mode for complex queries

### 4. **Production-Ready Design**
- Clean, modular architecture
- Type hints and comprehensive documentation
- Error handling and graceful degradation
- Minimal external dependencies

## 🚀 Quick Start

### Option 1: Kaggle Notebook (Recommended)

1. **Upload to Kaggle**
   - Create a new Kaggle notebook
   - Upload `notebooks/jarvis_kaggle_poc.py`
   - Or copy-paste the code directly

2. **Install Dependencies**
   ```python
   !pip install google-generativeai numpy requests
   ```

3. **Set API Key**
   ```python
   import os
   os.environ['GOOGLE_API_KEY'] = 'your-google-api-key-here'
   ```

   Or use Kaggle Secrets:
   - Go to Notebook Settings → Secrets
   - Add secret: `GOOGLE_API_KEY`
   - Access in code: `os.environ['GOOGLE_API_KEY']`

4. **Run the Demo**
   ```python
   %run notebooks/jarvis_kaggle_poc.py
   ```

### Option 2: Local Python Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/jarvis-assistant.git
   cd jarvis-assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r kaggle-requirements.txt
   ```

3. **Set API Key**
   ```bash
   export GOOGLE_API_KEY='your-google-api-key-here'
   ```

4. **Run the Demo**
   ```bash
   python notebooks/jarvis_kaggle_poc.py
   ```

## 📋 Requirements

### Essential
- **Python 3.8+**
- **Google API Key** (for Gemini) - [Get it here](https://makersuite.google.com/app/apikey)

### Dependencies
```
google-generativeai>=0.3.0  # Gemini API client
numpy>=1.24.0                # For embeddings
requests>=2.31.0             # For web search (optional)
```

## 🎮 Usage Modes

### 1. Automated Demo Mode (Default)
Demonstrates all capabilities with pre-defined scenarios:
```python
python notebooks/jarvis_kaggle_poc.py
# Select option 1
```

**Demo Scenarios:**
- ✅ Research & Information Gathering
- ✅ Data Analysis & Insights
- ✅ Memory-Aware Assistance
- ✅ Multi-Agent Collaboration

### 2. Interactive Mode
Chat directly with JARVIS:
```python
python notebooks/jarvis_kaggle_poc.py
# Select option 2
```

**Available Commands:**
- Type your question/task
- `multi` - Toggle multi-agent mode
- `stats` - View usage statistics
- `clear` - Clear memory
- `quit` - Exit

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              JARVIS Orchestrator                     │
│  (Intelligent Task Routing & Agent Coordination)     │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┴──────────┐
    │   Agent Selection  │ (Capability-based scoring)
    └────────┬──────────┘
             │
    ┌────────┴─────────────────────────────────┐
    │                                           │
┌───▼────────┐  ┌──────────────┐  ┌───────────▼─────┐
│ Researcher │  │ Data Analyst │  │ General Assistant │
│   Agent    │  │    Agent     │  │      Agent       │
└───┬────────┘  └──────┬───────┘  └───────┬───────┘
    │                  │                   │
    └──────────────────┴───────────────────┘
                       │
                  ┌────▼─────┐
                  │  Gemini  │ (LLM)
                  │   API    │
                  └────┬─────┘
                       │
              ┌────────┴─────────┐
              │                  │
         ┌────▼─────┐      ┌────▼─────┐
         │ Short-   │      │ Long-    │
         │ Term     │──────│ Term     │
         │ Memory   │      │ Memory   │
         └──────────┘      └──────────┘
              (Vector-based Semantic Search)
```

## 🎯 Demonstrated Capabilities

### 1. Multi-Modal LLM Integration ✅
- Advanced Gemini API integration
- Intelligent prompt engineering
- Context-aware generation
- Temperature control for different task types

### 2. Contextual Memory System ✅
- In-memory vector store with cosine similarity
- Semantic search for relevant past interactions
- Automatic memory consolidation
- Conversation context tracking

### 3. Multi-Agent Orchestration ✅
- **3 Specialized Agents** with distinct capabilities
- Intelligent task routing based on capability matching
- Confidence-based agent selection
- Multi-agent consensus mode

### 4. Adaptive Intelligence ✅
- Task-specific temperature tuning
- Agent specialization and expertise modeling
- Response synthesis from multiple agents
- Usage statistics and monitoring

## 📊 Example Outputs

### Research Task
```
📝 Task: What are the key differences between transformer and RNN architectures?
🤖 Agent: Researcher (confidence: 0.80)

💬 Response:
Key findings:
1. Architecture: Transformers use self-attention mechanisms, RNNs use sequential processing
2. Parallelization: Transformers can process sequences in parallel, RNNs are sequential
3. Long-range dependencies: Transformers handle better via attention
...
```

### Data Analysis Task
```
📝 Task: Analyze AI adoption trends in healthcare 2020-2024
🤖 Agent: DataAnalyst (confidence: 0.85)

💬 Response:
Analysis approach:
1. Methodology: Trend analysis with key indicators
2. Key insights:
   - 300% increase in AI diagnostic tools
   - Telemedicine AI integration accelerated post-2020
...
```

### Multi-Agent Collaboration
```
📝 Task: Should I invest in learning quantum computing for AI research?
🤖 Agent: JARVIS-MultiAgent (confidence: 0.80)

💬 Response:
Synthesized response from 3 agents:
[Combines research on quantum computing trends, analysis of job market data,
 and personalized recommendations based on your background]
```

## 🔧 Customization

### Add a New Agent

```python
class CustomAgent(BaseAgent):
    """Your custom agent."""

    def __init__(self, llm: GeminiClient, memory: MemoryManager):
        super().__init__("CustomAgent", llm, memory)
        self.capabilities = ["your", "custom", "keywords"]

    async def process(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        # Your implementation
        prompt = f"Custom processing for: {task}"
        response = await self.llm.generate(prompt)

        return AgentResponse(
            content=response,
            confidence=0.85,
            agent_name=self.name
        )

# Register in JarvisOrchestrator.__init__
self.agents["custom"] = CustomAgent(self.llm, self.memory)
```

### Modify Agent Capabilities

```python
# In agent __init__
self.capabilities = ["new", "capability", "keywords"]
```

### Adjust Memory Settings

```python
# Change short-term memory size (default: 10)
if len(self.short_term_memory) > 20:  # Keep last 20 instead of 10
    old = self.short_term_memory.pop(0)
    # ...
```

## 🎓 Educational Value

This POC demonstrates:

1. **Clean Architecture**: Separation of concerns, modularity, extensibility
2. **Design Patterns**:
   - Strategy Pattern (agent selection)
   - Factory Pattern (agent creation)
   - Observer Pattern (memory management)
3. **Production Best Practices**:
   - Type hints
   - Async/await patterns
   - Error handling
   - Configuration management
4. **AI/ML Concepts**:
   - Vector embeddings
   - Semantic search
   - Multi-agent systems
   - Prompt engineering

## 🚧 Limitations (POC Version)

**Not Included in POC:**
- ❌ Voice/Speech capabilities
- ❌ IoT device control
- ❌ Email/Calendar integration
- ❌ External databases (Redis, MongoDB, ChromaDB)
- ❌ Production API server (FastAPI)
- ❌ Containerization (Docker)

**For Full Version:**
See the main repository for production-ready version with all features.

## 📈 Performance

### Latency
- Single agent response: ~2-5 seconds
- Multi-agent synthesis: ~5-10 seconds
- Memory search: <100ms (in-memory)

### Resource Usage
- Memory: ~50-100MB (depending on conversation history)
- API calls: 1-3 per request (depending on mode)
- No GPU required

## 🔐 Security Notes

- **API Keys**: Never commit API keys to version control
- Use environment variables or Kaggle Secrets
- The POC doesn't store data persistently (all in-memory)
- No external network calls except to Google API (and optional web search)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a POC for educational purposes. For the full production system, see the main repository.

## 📚 Resources

- [Google AI Studio](https://makersuite.google.com/) - Get API keys
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Vector Embeddings Guide](https://ai.google.dev/docs/embeddings_guide)

## 💬 Support

For issues or questions:
1. Check the code comments (heavily documented)
2. Review demo output for examples
3. See main repository for full documentation

## ⭐ Credits

Built for the **Kaggle Agents Intensive Capstone Project**

Inspired by Iron Man's JARVIS - *"Just A Rather Very Intelligent System"*

---

**Ready to run?** 🚀

```python
# Set your API key
import os
os.environ['GOOGLE_API_KEY'] = 'your-key-here'

# Run the demo
%run notebooks/jarvis_kaggle_poc.py
```

**Happy coding!** 🎉
