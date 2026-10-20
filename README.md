# AgentLite

A multi-agent task automation system that runs on small, locally hosted LLMs
(Ollama on a 4 GB GTX 1650), with optional Azure OpenAI and Gemini providers.

> Work in progress. The full README (architecture diagram, eval results, design
> decisions) arrives in Phase 7.

## Quick start (Windows, PowerShell)

```powershell
# 1. Ollama (native Windows app): https://ollama.com/download
ollama pull qwen2.5:3b

# 2. Python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# 3. Run a task
python -m agentlite "What is LangGraph and who maintains it?"
python -m agentlite --list-models
```

## Layout so far

| Path | Purpose |
|---|---|
| `agentlite/config.py` | All settings, read from `.env` |
| `agentlite/llm.py` | Provider-agnostic LLM layer (`LLM.invoke` returns message + tokens + latency) |
| `agentlite/tools/` | One tool per file; the docstring is what the model sees |
| `agentlite/agents/tool_agent.py` | Tool-calling loop with recovery for small-model mistakes |
| `agentlite/__main__.py` | CLI |
