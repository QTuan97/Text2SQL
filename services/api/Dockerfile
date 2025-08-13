FROM python:3.12-slim

ENV POETRY_VERSION=1.8.3 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential jq && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]