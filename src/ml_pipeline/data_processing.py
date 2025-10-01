# ml_pipeline/data_processing.py
import pandas as pd
import json
import os
from ml_pipeline import config

def _load_keyed_json(file_path, id_column_name, nested_list_key=None):
    # ... (código completo da função _load_keyed_json)
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        records = []
        for key, value in data.items():
            if nested_list_key:
                for item in value.get(nested_list_key, []):
                    item[id_column_name] = key
                    records.append(item)
            else:
                value[id_column_name] = key
                records.append(value)
        return pd.DataFrame(records)
    except Exception as e: return pd.DataFrame()

def _flatten_nested_columns(df):
    # ... (código completo da função _flatten_nested_columns)
    df_out = df.copy()
    for col in df.columns:
        first_item = df_out[col].dropna().iloc[0] if not df_out[col].dropna().empty else None
        if isinstance(first_item, dict):
            normalized_df = pd.json_normalize(df_out[col]).add_prefix(f'{col}.')
            df_out = df_out.drop(columns=[col]).join(normalized_df)
    return df_out

def create_unified_dataset():
    """Cria o dataset unificado a partir dos JSONs brutos."""
    print("Iniciando pré-processamento dos dados brutos (JSONs)...")
    df_jobs = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'vagas.json'), 'vaga_id') # Usando vagas.json como no arquivo original
    df_applicants = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'applicants.json'), 'applicant_id')
    df_prospects = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'prospects.json'), 'vaga_id', nested_list_key='prospects')
    
    if any(df.empty for df in [df_jobs, df_applicants, df_prospects]):
        raise RuntimeError("Falha ao carregar um ou mais arquivos JSON brutos.")

    df_jobs = _flatten_nested_columns(df_jobs)
    df_applicants = _flatten_nested_columns(df_applicants)

    df_merged = pd.merge(df_prospects, df_applicants, left_on='codigo', right_on='applicant_id', how='inner')
    df_final = pd.merge(df_merged, df_jobs, on='vaga_id', how='inner', suffixes=('_candidato', '_vaga'))

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df_final.to_parquet(config.UNIFIED_DATA_PATH, index=False)
    print(f"Dados unificados salvos com sucesso em '{config.UNIFIED_DATA_PATH}'")

if __name__ == "__main__":
    create_unified_dataset()