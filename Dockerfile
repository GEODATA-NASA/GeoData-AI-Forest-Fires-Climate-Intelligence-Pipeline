# GeoData AI - Forest Fires & Climate Intelligence Pipeline
# One image serves both the API (default) and the pipeline (`python main.py`).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libraries needed by psycopg2-binary / numpy wheels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Data, model artifacts and logs are mounted as volumes (see docker-compose.yml).
RUN mkdir -p raw_data cleaned_data logs modeling/artifacts \
    && useradd --create-home --uid 1000 geodata \
    && chown -R geodata:geodata /app
USER geodata

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
