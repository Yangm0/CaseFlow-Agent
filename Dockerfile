FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANGGRAPH_STRICT_MSGPACK=true \
    CASEFLOW_WEB_ROOT=/app/web

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
RUN python -m pip install --no-cache-dir .

EXPOSE 8000 8001
