# main.py (versão com a função predict CORRIGIDA)

from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import joblib
import pandas as pd
import pathlib

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Match de Vagas V2",
    description="Uma API com um modelo otimizado para prever a compatibilidade.",
    version="2.0"
)

# --- Carregando o pipeline completo (modelo V2) ---
try:
    PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent.parent
    PIPELINE_PATH = PROJECT_ROOT / "pipeline_model_v2.joblib.gz" # <-- Correto
    
    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline do Modelo V2 carregada com sucesso.")

except FileNotFoundError:
    print(f"ERRO: Pipeline não encontrada no caminho: {PIPELINE_PATH}")
    pipeline = None

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(response: Response):
    """
    Verifica se a API está no ar e se o modelo de ML foi carregado corretamente.
    - Retorna 200 OK se o modelo estiver carregado.
    - Retorna 503 Service Unavailable se o modelo falhou ao carregar.
    """
    if not pipeline:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "detail": "Modelo de Machine Learning não foi carregado."}
    
    return {"status": "healthy", "detail": "API e modelo de ML estão operacionais."}


# --- Definindo o formato de entrada dos dados (mais colunas) ---
class CandidateData(BaseModel):
    nivel_profissional_candidato: str
    nivel_academico_candidato: str
    nivel_ingles_candidato: str
    nivel_profissional_vaga: str
    nivel_academico_vaga: str
    nivel_ingles_vaga: str
    area_atuacao_candidato: str
    area_atuacao_vaga: str
    tipo_contratacao_vaga: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "nivel_profissional_candidato": "sênior",
                "nivel_academico_candidato": "ensino superior completo",
                "nivel_ingles_candidato": "avançado",
                "nivel_profissional_vaga": "sênior",
                "nivel_academico_vaga": "ensino superior completo",
                "nivel_ingles_vaga": "fluente",
                "area_atuacao_candidato": "ti - sistemas e ferramentas-",
                "area_atuacao_vaga": "ti - sistemas e ferramentas-",
                "tipo_contratacao_vaga": "clt full"
            }
        }

# --- Criando o endpoint de predição (VERSÃO CORRIGIDA) ---
@app.post("/predict")
def predict(data: CandidateData):
    if not pipeline:
        return {"error": "Pipeline do modelo não foi carregada. Verifique os logs."}

    # 1. Converter os dados de entrada para um DataFrame
    # A ordem das colunas é importante para a etapa de criação das features de match
    input_data = {
        'informacoes_profissionais.nivel_profissional': [data.nivel_profissional_candidato],
        'formacao_e_idiomas.nivel_academico': [data.nivel_academico_candidato],
        'formacao_e_idiomas.nivel_ingles': [data.nivel_ingles_candidato],
        'perfil_vaga.nivel profissional': [data.nivel_profissional_vaga],
        'perfil_vaga.nivel_academico': [data.nivel_academico_vaga], # << Erro de digitação corrigido aqui
        'perfil_vaga.nivel_ingles': [data.nivel_ingles_vaga],
        'informacoes_profissionais.area_atuacao': [data.area_atuacao_candidato],
        'perfil_vaga.areas_atuacao': [data.area_atuacao_vaga],
        'informacoes_basicas.tipo_contratacao': [data.tipo_contratacao_vaga]
    }
    df = pd.DataFrame(input_data)

    # 2. <<< ETAPA FALTANTE ADICIONADA AQUI >>>
    # Criar as features de 'match_*' exatamente como fizemos no treinamento
    df['match_nivel_profissional'] = (df['informacoes_profissionais.nivel_profissional'].str.lower() == df['perfil_vaga.nivel profissional'].str.lower()).astype(int)
    df['match_nivel_academico'] = (df['formacao_e_idiomas.nivel_academico'].str.lower() == df['perfil_vaga.nivel_academico'].str.lower()).astype(int)
    df['match_nivel_ingles'] = (df['formacao_e_idiomas.nivel_ingles'].str.lower() == df['perfil_vaga.nivel_ingles'].str.lower()).astype(int)
    df['match_area_atuacao'] = (df['informacoes_profissionais.area_atuacao'].str.lower() == df['perfil_vaga.areas_atuacao'].str.lower()).astype(int)
    
    # 3. Fazer a predição usando a pipeline completa
    # Agora o df tem todas as colunas que a pipeline espera
    prediction = pipeline.predict(df)
    prediction_proba = pipeline.predict_proba(df)

    # 4. Retornar o resultado
    return {
        "prediction": int(prediction[0]),
        "probability_no_match": f"{prediction_proba[0][0]:.4f}",
        "probability_match": f"{prediction_proba[0][1]:.4f}"
    }

@app.get("/")
def read_root():
    return {"status": "API de Match de Vagas V2 está no ar!"}