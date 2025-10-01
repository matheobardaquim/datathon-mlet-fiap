import joblib
import pandas as pd
from datetime import datetime, timezone
import pathlib
import logging
import json
from src.schemas.candidate import CandidateData

project_root = pathlib.Path(__file__).parent.parent
log_path = project_root / "predictions.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename=str(log_path), 
    force=True             
)

logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        try:
            project_root_model = pathlib.Path(__file__).parent.parent
            pipeline_path = project_root_model / "models" / "pipeline_model_v2.joblib.gz"
            
            pipeline = joblib.load(pipeline_path)
            print(f"Pipeline carregada com sucesso de: {pipeline_path}")
            return pipeline
        except FileNotFoundError:
            print(f"ERRO: Pipeline não encontrada no caminho: {pipeline_path}")
            return None

    def is_ready(self):
        return self.pipeline is not None

    def predict(self, data: CandidateData) -> dict:
        if not self.is_ready():
            return {"error": "Pipeline do modelo não foi carregada."}

        df = pd.DataFrame([data.model_dump()])
        df.columns = [
            'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
            'formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel profissional',
            'perfil_vaga.nivel_academico', 'perfil_vaga.nivel_ingles',
            'informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao',
            'informacoes_basicas.tipo_contratacao'
        ]

        df['match_nivel_profissional'] = (df['informacoes_profissionais.nivel_profissional'].str.lower() == df['perfil_vaga.nivel profissional'].str.lower()).astype(int)
        df['match_nivel_academico'] = (df['formacao_e_idiomas.nivel_academico'].str.lower() == df['perfil_vaga.nivel_academico'].str.lower()).astype(int)
        df['match_nivel_ingles'] = (df['formacao_e_idiomas.nivel_ingles'].str.lower() == df['perfil_vaga.nivel_ingles'].str.lower()).astype(int)
        df['match_area_atuacao'] = (df['informacoes_profissionais.area_atuacao'].str.lower() == df['perfil_vaga.areas_atuacao'].str.lower()).astype(int)
        
        prediction = self.pipeline.predict(df)
        prediction_proba = self.pipeline.predict_proba(df)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": data.model_dump(),
            "prediction": int(prediction[0]),
            "probability_match": float(prediction_proba[0][1])
        }
        logger.info(json.dumps(log_entry))
        
        return {
            "prediction": int(prediction[0]),
            "probability_no_match": f"{prediction_proba[0][0]:.4f}",
            "probability_match": f"{prediction_proba[0][1]:.4f}"
        }

ml_service = MLService()