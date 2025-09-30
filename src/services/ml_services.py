# app/services/ml_service.py

import joblib
import pandas as pd
import pathlib
from src.schemas.candidate import CandidateData

class MLService:
    def __init__(self):
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Carrega a pipeline de ML a partir do arquivo .joblib.gz."""
        try:
            # O caminho é relativo à raiz do projeto, que será /app no Docker
            pipeline_path = pathlib.Path(__file__).parent.parent.parent / "pipeline_model_v2.joblib.gz"
            pipeline = joblib.load(pipeline_path)
            print("Pipeline do Modelo V2 carregada com sucesso.")
            return pipeline
        except FileNotFoundError:
            print(f"ERRO: Pipeline não encontrada no caminho: {pipeline_path}")
            return None

    def is_ready(self):
        """Verifica se a pipeline foi carregada."""
        return self.pipeline is not None

    def predict(self, data: CandidateData) -> dict:
        """Executa a predição usando a pipeline carregada."""
        if not self.is_ready():
            return {"error": "Pipeline do modelo não foi carregada."}

        # 1. Converte os dados de entrada para um DataFrame
        df = pd.DataFrame([data.model_dump()])
        # Renomeia as colunas para o formato esperado pelo modelo
        df.columns = [
            'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
            'formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel profissional',
            'perfil_vaga.nivel_academico', 'perfil_vaga.nivel_ingles',
            'informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao',
            'informacoes_basicas.tipo_contratacao'
        ]

        # 2. Cria as features de 'match_*'
        df['match_nivel_profissional'] = (df['informacoes_profissionais.nivel_profissional'].str.lower() == df['perfil_vaga.nivel profissional'].str.lower()).astype(int)
        df['match_nivel_academico'] = (df['formacao_e_idiomas.nivel_academico'].str.lower() == df['perfil_vaga.nivel_academico'].str.lower()).astype(int)
        df['match_nivel_ingles'] = (df['formacao_e_idiomas.nivel_ingles'].str.lower() == df['perfil_vaga.nivel_ingles'].str.lower()).astype(int)
        df['match_area_atuacao'] = (df['informacoes_profissionais.area_atuacao'].str.lower() == df['perfil_vaga.areas_atuacao'].str.lower()).astype(int)
        
        # 3. Faz a predição
        prediction = self.pipeline.predict(df)
        prediction_proba = self.pipeline.predict_proba(df)

        # 4. Retorna o resultado
        return {
            "prediction": int(prediction[0]),
            "probability_no_match": f"{prediction_proba[0][0]:.4f}",
            "probability_match": f"{prediction_proba[0][1]:.4f}"
        }

# Cria uma instância única do serviço que será usada em toda a aplicação
ml_service = MLService()