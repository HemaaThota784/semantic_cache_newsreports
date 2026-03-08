FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer so it's cached on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY api/        ./api/
COPY analysis/   ./analysis/
COPY cache/      ./cache/
COPY embeddings/ ./embeddings/
COPY .env.example .env

# These are mounted at runtime via docker-compose volumes
# chroma_db/ and models/ must exist on the host before running
# (run python scripts/build_index.py first)

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
