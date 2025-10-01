# tests/test_ml_service.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Importa as classes que vamos testar
# ATENÇÃO: o nome do seu arquivo é ml_services.py (plural)
from src.services.ml_services import MLService 
from src.schemas.candidate import CandidateData

def test_ml_service_model_loads_successfully(mocker):
    """Testa se o serviço é inicializado corretamente quando o modelo é encontrado."""
    mocker.patch('joblib.load', return_value=MagicMock())
    ml_service = MLService()
    assert ml_service.is_ready() is True

def test_ml_service_handles_model_not_found(mocker):
    """Testa se o serviço lida com o erro de arquivo não encontrado."""
    mocker.patch('joblib.load', side_effect=FileNotFoundError)
    ml_service = MLService()
    assert ml_service.is_ready() is False

def test_predict_logic_and_feature_engineering(mocker):
    """Testa a lógica da função 'predict' e a criação de features."""
    # Simula o carregamento do modelo na inicialização
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([1])
    mock_pipeline.predict_proba.return_value = np.array([[0.25, 0.75]])
    mocker.patch('joblib.load', return_value=mock_pipeline)

    ml_service = MLService()

    # Cria dados de entrada
    sample_data = CandidateData(
        nivel_profissional_candidato="sênior", nivel_academico_candidato="ensino superior completo",
        nivel_ingles_candidato="avançado", nivel_profissional_vaga="sênior",
        nivel_academico_vaga="ensino superior completo", nivel_ingles_vaga="avançado",
        area_atuacao_candidato="ti", area_atuacao_vaga="ti",
        tipo_contratacao_vaga="clt full"
    )

    result = ml_service.predict(sample_data)

    # Verifica se a saída está correta
    assert result['prediction'] == 1
    assert result['probability_match'] == "0.7500"

    # Verifica se o modelo interno foi chamado corretamente
    ml_service.pipeline.predict.assert_called_once()
    
    # Captura o DataFrame que foi passado para o modelo
    df_passed_to_model = ml_service.pipeline.predict.call_args[0][0]

    # Verifica se a engenharia de features funcionou
    assert 'match_nivel_profissional' in df_passed_to_model.columns
    assert df_passed_to_model['match_nivel_profissional'].iloc[0] == 1