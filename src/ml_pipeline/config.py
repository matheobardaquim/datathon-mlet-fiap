RAW_DATA_DIR = "Data/raw/"
PROCESSED_DATA_DIR = "Data/processed/"
MODELS_DIR = "models/"
UNIFIED_DATA_PATH = PROCESSED_DATA_DIR + "dados_unificados.parquet"
MODEL_PATH = MODELS_DIR + "pipeline_model_v2.joblib.gz"
STATUS_POSITIVOS = [
    'Contratado pela Decision', 'Contratado como Hunting', 'Encaminhado ao Requisitante',
    'Entrevista Técnica', 'Entrevista com Cliente', 'Aprovado', 'Em avaliação pelo RH',
    'Documentação PJ', 'Documentação CLT', 'Documentação Cooperado',
    'Encaminhar Proposta', 'Proposta Aceita'
]
TARGET_COLUMN = 'situacao_candidado'
CATEGORICAL_FEATURES = [
    'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel profissional',
    'perfil_vaga.nivel_academico', 'perfil_vaga.nivel_ingles',
    'informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao',
    'informacoes_basicas.tipo_contratacao'
]
MATCH_FEATURES = [
    'match_nivel_profissional', 'match_nivel_academico', 
    'match_nivel_ingles', 'match_area_atuacao'
]
MODEL_PARAMS = {
    'n_estimators': 150, 'random_state': 42,
    'class_weight': 'balanced', 'n_jobs': -1
}