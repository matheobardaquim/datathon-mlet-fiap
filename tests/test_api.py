# tests/test_api.py

from fastapi.testclient import TestClient
import sys
import os
import pathlib
import importlib.util

main_path = pathlib.Path(__file__).parent.parent / "src" / "clustering-model" / "main.py"
spec = importlib.util.spec_from_file_location("main", main_path)
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)
app = main.app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API de Match de Vagas V2 está no ar!"}

def test_predict_match():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert "probability_match" in response_json

def test_predict_invalid_payload():
    # Teste para verificar se a validação do Pydantic funciona
    payload = {
        "nivel_profissional_candidato": "sênior"
        # Faltando os outros campos
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # 422 Unprocessable Entity é o erro do FastAPI