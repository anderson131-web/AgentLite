# AgentLite app image: FastAPI backend + built React frontend in ONE container.
# One container keeps local docker-compose and Cloud Run / Azure Container Apps
# identical: a single service on a single port, no CORS, no reverse proxy.
#
# Ollama is NOT in this image. Locally the app reaches Ollama on the host via
# host.docker.internal; in the cloud it uses Gemini or Azure OpenAI instead.

# ---- Stage 1: build the React app ------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime ------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080
WORKDIR /app

# Install dependencies first so code changes don't invalidate this cached layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentlite/ agentlite/
COPY api/ api/
COPY evals/ evals/
COPY examples/ examples/
COPY docs/sample_resume.md docs/
COPY --from=frontend /build/dist frontend/dist

# Run as a non-root user; only data/, docs/ and workspace/ are writable.
RUN useradd --create-home app \
    && mkdir -p data workspace \
    && chown -R app:app data docs workspace
USER app

EXPOSE 8080
# Cloud Run and Container Apps set $PORT; default 8080 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
