# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import pathlib

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Match de Vagas",
    description="Uma API para prever a compatibilidade entre candidatos e vagas.",
    version="1.0"
)

# --- Carregando os modelos ---
# Estes arquivos foram criados no Passo 2 e 3
try:
    # Volta duas pastas para chegar na raiz do projeto
    PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent.parent 
    
    # Constrói o caminho correto para a pasta /data/
    PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "preprocessor.joblib"
    MODEL_PATH = PROJECT_ROOT / "data" / "random_forest_model.joblib"
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("Modelos carregados com sucesso a partir de:", PROJECT_ROOT / "data")

except FileNotFoundError:
    print(f"ERRO: Arquivos de modelo não encontrados nos caminhos esperados:")
    print(f"Tentativa de caminho para o Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Tentativa de caminho para o Model: {MODEL_PATH}")
    preprocessor = None
    model = None

# --- Definindo o formato de entrada dos dados ---
# Usamos Pydantic para validar os dados que chegam na API.
# As informações devem ser exatamente as mesmas que usamos para treinar o modelo.
class CandidateData(BaseModel):
    # Features do candidato
    nivel_profissional_candidato: str
    nivel_academico_candidato: str
    nivel_ingles_candidato: str
    # Features da vaga
    nivel_profissional_vaga: str
    nivel_academico_vaga: str
    nivel_ingles_vaga: str

    class Config:
        json_schema_extra  = {
            "example": {
                "nivel_profissional_candidato": "sênior",
                "nivel_academico_candidato": "ensino superior completo",
                "nivel_ingles_candidato": "avançado",
                "nivel_profissional_vaga": "sênior",
                "nivel_academico_vaga": "ensino superior completo",
                "nivel_ingles_vaga": "fluente"
            }
        }

# --- Criando o endpoint de predição ---
@app.post("/predict")
def predict(data: CandidateData):
    """
    Recebe os dados de um candidato e de uma vaga e retorna a predição de match.
    - **0**: Não-Match
    - **1**: Match
    """
    if not preprocessor or not model:
        return {"error": "Modelos não foram carregados. Verifique os logs do servidor."}

    # 1. Converter os dados de entrada para um DataFrame do Pandas
    # A ordem das colunas DEVE ser a mesma do treinamento.
    input_data = {
        'informacoes_profissionais.nivel_profissional': [data.nivel_profissional_candidato.lower()],
        'formacao_e_idiomas.nivel_academico': [data.nivel_academico_candidato.lower()],
        'formacao_e_idiomas.nivel_ingles': [data.nivel_ingles_candidato.lower()],
        'perfil_vaga.nivel profissional': [data.nivel_profissional_vaga.lower()],
        'perfil_vaga.nivel_academico': [data.nivel_academico_vaga.lower()],
        'perfil_vaga.nivel_ingles': [data.nivel_ingles_vaga.lower()]
    }
    df = pd.DataFrame(input_data)
    
    # 2. Criar as features de 'match_*' da mesma forma que no treino
    df['match_nivel_profissional'] = (df[df.columns[0]] == df[df.columns[3]]).astype(int)
    df['match_nivel_academico'] = (df[df.columns[1]] == df[df.columns[4]]).astype(int)
    df['match_nivel_ingles'] = (df[df.columns[2]] == df[df.columns[5]]).astype(int)

    # 3. Aplicar o pré-processamento salvo
    processed_data = preprocessor.transform(df)

    # 4. Fazer a predição
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)

    # 5. Retornar o resultado
    return {
        "prediction": int(prediction[0]),
        "probability_no_match": f"{prediction_proba[0][0]:.4f}",
        "probability_match": f"{prediction_proba[0][1]:.4f}"
    }

# --- Endpoint raiz para verificar se a API está no ar ---
@app.get("/")
def read_root():
    return {"status": "API de Match de Vagas está no ar!"}