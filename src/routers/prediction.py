from fastapi import APIRouter, Response, status
from src.schemas.candidate import CandidateData
from src.services.ml_services import ml_service

router = APIRouter()

@router.get("/", tags=["General"])
def read_root():
    return {"status": "API de Match de Vagas V2 está no ar!"}

@router.get("/health", tags=["Health Check"], status_code=status.HTTP_200_OK)
def health_check(response: Response):
    """
    Verifica a saúde da API e do modelo de ML.
    """
    if not ml_service.is_ready():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "detail": "Modelo de Machine Learning não foi carregado."}
    
    return {"status": "healthy", "detail": "API e modelo de ML estão operacionais."}

@router.post("/predict", tags=["Prediction"])
def predict(data: CandidateData):
    """
    Recebe dados de um candidato e vaga e retorna a predição de match.
    """
    prediction_result = ml_service.predict(data)
    if "error" in prediction_result:
        return Response(content=prediction_result["error"], status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return prediction_result