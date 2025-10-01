# ml_pipeline/data_processing.py

import pandas as pd
import json
import os
import zipfile
from ml_pipeline import config

def unzip_raw_data_if_needed():
    print("Verificando a necessidade de descompactar dados brutos...")
    raw_files = ['applicants', 'prospects', 'vagas']
    for file_name in raw_files:
        json_path = os.path.join(config.RAW_DATA_DIR, file_name)
        zip_path = json_path + '.zip'
        if not os.path.exists(json_path) and os.path.exists(zip_path):
            print(f"Arquivo '{file_name}' não encontrado. Descompactando de '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.RAW_DATA_DIR)

def _load_keyed_json(file_path, id_column_name, nested_list_key=None):
    """
    Função auxiliar para carregar um JSON. 
    REMOVEMOS o try-except para que o erro real apareça.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
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

def _flatten_nested_columns(df):
    df_out = df.copy()
    for col in df.columns:
        first_item = df_out[col].dropna().iloc[0] if not df_out[col].dropna().empty else None
        if isinstance(first_item, dict):
            print(f"Achatando coluna aninhada: '{col}'")
            normalized_df = pd.json_normalize(df_out[col]).add_prefix(f'{col}.')
            df_out = df_out.drop(columns=[col]).join(normalized_df)
    return df_out

def create_unified_dataset():
    """
    Executa o processo de carga, junção e limpeza dos dados brutos.
    """
    unzip_raw_data_if_needed()
    print("Iniciando pré-processamento dos dados brutos (JSONs)...")

    # Adicionando prints para rastrear cada passo
    print("Carregando vagas.json...")
    df_jobs = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'vagas.json'), 'vaga_id')
    print("Carregando applicants.json...")
    df_applicants = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'applicants.json'), 'applicant_id')
    print("Carregando prospects.json...")
    df_prospects = _load_keyed_json(os.path.join(config.RAW_DATA_DIR, 'prospects.json'), 'vaga_id', nested_list_key='prospects')
    
    if any(df.empty for df in [df_jobs, df_applicants, df_prospects]):
        raise RuntimeError("Falha ao carregar um ou mais arquivos JSON brutos.")

    print("Achatando colunas aninhadas...")
    df_jobs = _flatten_nested_columns(df_jobs)
    df_applicants = _flatten_nested_columns(df_applicants)

    print("Fazendo merge de prospects e applicants...")
    df_merged = pd.merge(df_prospects, df_applicants, left_on='codigo', right_on='applicant_id', how='inner')
    print("Fazendo merge final com vagas...")
    df_final = pd.merge(df_merged, df_jobs, on='vaga_id', how='inner', suffixes=('_candidato', '_vaga'))

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df_final.to_parquet(config.UNIFIED_DATA_PATH, index=False)
    print(f"Dados unificados salvos com sucesso em '{config.UNIFIED_DATA_PATH}'")

if __name__ == "__main__":
    create_unified_dataset()