# app/schemas/candidate.py

from pydantic import BaseModel, ConfigDict

class CandidateData(BaseModel):
    """
    Schema de dados para a entrada da predição.
    Valida os tipos e a estrutura dos dados recebidos pela API.
    """
    nivel_profissional_candidato: str
    nivel_academico_candidato: str
    nivel_ingles_candidato: str
    nivel_profissional_vaga: str
    nivel_academico_vaga: str
    nivel_ingles_vaga: str
    area_atuacao_candidato: str
    area_atuacao_vaga: str
    tipo_contratacao_vaga: str
    
    model_config = ConfigDict(
        json_schema_extra={
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
    )