# app/main.py

from fastapi import FastAPI
from src.routers import prediction, prometheus

app = FastAPI(
    title="API de Match de Vagas V2",
    description="Uma API com um modelo otimizado para prever a compatibilidade.",
    version="2.0"
)

# Inclui as rotas definidas no arquivo prediction.py
app.include_router(prediction.router)
app.include_router(prometheus.router)