# Jarvis AI Personal Assistant

Production-ready AI personal assistant leveraging Google ADK, Gemini, and multi-agent architecture. Built for the Kaggle Agents Intensive Capstone Project.

## Features

- **Multi-Modal Understanding**: Voice, text, and vision processing
- **Contextual Memory**: ChromaDB-powered short/long-term memory with semantic search
- **Multi-Agent System**: Specialized agents for different tasks (scheduling, research, analysis)
- **LLM Integration**: Gemini Pro with fallback to OpenAI/Claude
- **Production-Ready**: FastAPI, Docker, monitoring, testing

## Quick Start

```bash
# Clone repository
git clone https://github.com/amlanacharya/google-agent-kaggle-jarvis.git
cd google-agent-kaggle-jarvis

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start services with Docker
docker-compose up -d

# Or run locally
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Jarvis Main Agent             │
│  (Orchestrator & Task Router)           │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼───────┐
│  Specialized   │  │  Specialized │
│    Agents      │  │    Agents    │
└────────────────┘  └──────────────┘

Core Components:
├── LLM Layer (Gemini/OpenAI/Claude)
├── Memory Manager (ChromaDB)
├── Agent Framework (BaseAgent)
└── API Layer (FastAPI)
```

## Project Structure

```
jarvis-assistant/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── base_agent.py
│   │   └── jarvis_agent.py
│   ├── core/            # Core functionality
│   │   ├── config.py    # Settings management
│   │   ├── logger.py    # Logging
│   │   ├── llm.py       # LLM integration
│   │   └── memory.py    # Memory system
│   ├── api/             # API endpoints
│   │   └── health.py
│   ├── integrations/    # External integrations
│   └── main.py          # Entry point
├── tests/               # Test suite
├── deployment/          # Deployment configs
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Configuration

Required environment variables in `.env`:

```bash
# Google Gemini
GOOGLE_API_KEY=your_key_here

# Optional: Alternative LLMs
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Services (auto-configured with Docker)
CHROMADB_HOST=localhost
REDIS_HOST=localhost
MONGODB_URI=mongodb://localhost:27017
```

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

## Development

```bash
# Run tests
pytest tests/ --cov=src

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

## Technology Stack

- **AI/ML**: Google Gemini, LangChain, ChromaDB
- **Backend**: FastAPI, Python 3.11+
- **Storage**: Redis, MongoDB, ChromaDB
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Docker, Kubernetes-ready

## Roadmap

See [progress.md](progress.md) for detailed development log.

- [x] Phase 1: Core Setup
- [ ] Phase 2: Integration Layer (Google ADK, MCP)
- [ ] Phase 3: Prototype with voice/task execution
- [ ] Phase 4: Advanced features (multi-agent, analytics)
- [ ] Phase 5: Production deployment

## License

MIT

## Contact

Built by amlanacharya for Kaggle Agents Intensive Capstone Project
