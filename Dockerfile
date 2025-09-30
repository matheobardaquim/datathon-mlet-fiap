# =================================================================
# Estágio 1: Base e Instalação de Dependências
# =================================================================

# Usar uma imagem base oficial do Python. A versão 'slim' é mais leve.
FROM python:3.11-slim

# Definir variáveis de ambiente no formato moderno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar o arquivo de dependências (que está na raiz do projeto)
COPY requirements.txt .

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# =================================================================
# Estágio 2: Copiar a Aplicação e Executar
# =================================================================

# Copiar os artefatos e o código-fonte para o contêiner
# A CORREÇÃO FINAL ESTÁ AQUI:
# Copia o modelo de ML de dentro da pasta 'Data' para a raiz do WORKDIR (/app)
COPY Data/pipeline_model_v2.joblib .

# Copia toda a pasta 'src' para dentro do WORKDIR, mantendo sua estrutura
COPY src/ ./src

# Expor a porta em que a aplicação vai rodar.
EXPOSE 8000

# Comando para iniciar a aplicação.
CMD ["uvicorn", "src.clustering-model.main:app", "--host", "0.0.0.0", "--port", "8000"]