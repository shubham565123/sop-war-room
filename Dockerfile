FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py stakeholders.py ./
COPY server/ ./server/

EXPOSE 8000

CMD ["python", "-m", "server.app"]
