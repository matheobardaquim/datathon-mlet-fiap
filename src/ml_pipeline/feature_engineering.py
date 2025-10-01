# ml_pipeline/feature_engineering.py
import pandas as pd
import numpy as np
from ml_pipeline import config

def create_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de 'match'."""
    print("Executando a engenharia de features...")
    df_out = df.copy()
    comparisons = {
        'match_nivel_profissional': ('informacoes_profissionais.nivel_profissional', 'perfil_vaga.nivel profissional'),
        'match_nivel_academico': ('formacao_e_idiomas.nivel_academico', 'perfil_vaga.nivel_academico'),
        'match_nivel_ingles': ('formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel_ingles'),
        'match_area_atuacao': ('informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao')
    }
    all_cols_to_clean = [col for pair in comparisons.values() for col in pair] + ['informacoes_basicas.tipo_contratacao']
    for col in all_cols_to_clean:
        if col in df_out.columns: df_out[col] = df_out[col].fillna('N/A').str.lower()
    for new_feature, (col1, col2) in comparisons.items():
        df_out[new_feature] = np.where(df_out[col1] == df_out[col2], 1, 0)
    return df_out