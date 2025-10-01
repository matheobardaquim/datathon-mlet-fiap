from fastapi.testclient import TestClient
from src.main import app
from src.routers import prediction  

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API de Match de Vagas V2 está no ar!"}

def test_health_check_when_healthy(mocker):
    mocker.patch.object(prediction.ml_service, 'is_ready', return_value=True)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "detail": "API e modelo de ML estão operacionais."}

def test_health_check_when_unhealthy(mocker):
    mocker.patch.object(prediction.ml_service, 'is_ready', return_value=False)
    
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy", "detail": "Modelo de Machine Learning não foi carregado."}

def test_predict_endpoint(mocker):
    mock_prediction = {
        "prediction": 1, "probability_no_match": "0.1000", "probability_match": "0.9000"
    }
    mocker.patch.object(prediction.ml_service, 'predict', return_value=mock_prediction)

    payload = {
        "nivel_profissional_candidato": "sênior", "nivel_academico_candidato": "ensino superior completo",
        "nivel_ingles_candidato": "avançado", "nivel_profissional_vaga": "sênior",
        "nivel_academico_vaga": "ensino superior completo", "nivel_ingles_vaga": "fluente",
        "area_atuacao_candidato": "ti", "area_atuacao_vaga": "ti",
        "tipo_contratacao_vaga": "clt full"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == mock_prediction