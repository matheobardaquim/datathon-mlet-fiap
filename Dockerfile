FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# instalar curl para HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copia o modelo comprimido para a raiz do WORKDIR
COPY src/models/pipeline_model_v2.joblib.gz .

# Copia toda a nova pasta 'src' para dentro do WORKDIR
COPY src/ ./src

EXPOSE 8000

# healthcheck: verifica o endpoint /health (ajuste a URL se necessário)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]