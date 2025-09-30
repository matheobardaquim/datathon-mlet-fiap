# Dockerfile ATUALIZADO

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o modelo comprimido para a raiz do WORKDIR (/app)
COPY Data/pipeline_model_v2.joblib.gz .

# Copia toda a nova pasta 'src' para dentro do WORKDIR
COPY src/ ./src

EXPOSE 8000

# ATUALIZAÇÃO: O comando agora aponta para o novo 'main.py' dentro da pasta 'app'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]