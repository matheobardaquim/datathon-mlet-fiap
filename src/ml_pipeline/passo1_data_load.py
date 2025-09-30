# passo1_data_load_v4.py

import pandas as pd
import json

def load_keyed_json_to_dataframe(file_path, id_column_name, nested_list_key=None):
    """Carrega um JSON estruturado como um dicionário (chaveado por ID) e o converte em um DataFrame."""
    print(f"Carregando e processando '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for key, value in data.items():
            if nested_list_key:
                nested_items = value.get(nested_list_key, [])
                for item in nested_items:
                    item[id_column_name] = key
                    records.append(item)
            else:
                value[id_column_name] = key
                records.append(value)
        
        df = pd.DataFrame(records)
        print(f"Arquivo '{file_path}' carregado com sucesso. {len(df)} registros processados.")
        return df
    except Exception as e:
        print(f"Ocorreu um erro ao processar {file_path}: {e}")
        return pd.DataFrame()

def flatten_nested_columns(df):
    """Expande colunas que contêm dicionários ou listas de dicionários."""
    df_out = df.copy()
    for col in df.columns:
        if isinstance(df_out[col].iloc[0], dict):
            print(f"Achatando coluna aninhada: '{col}'")
            # Expande a coluna de dicionários em novas colunas
            normalized_df = pd.json_normalize(df_out[col]).add_prefix(f'{col}.')
            # Remove a coluna original e junta com a nova achatada
            df_out = df_out.drop(columns=[col]).join(normalized_df)
    return df_out

# --- 1. Carregando os Dados com a Lógica Correta ---
df_jobs = load_keyed_json_to_dataframe('vagas.json', id_column_name='vaga_id')
df_applicants = load_keyed_json_to_dataframe('Applicants.json', id_column_name='applicant_id')
# Usando as chaves corretas que você forneceu no exemplo
df_prospects = load_keyed_json_to_dataframe('Prospects.json', id_column_name='vaga_id', nested_list_key='prospects')

if df_jobs.empty or df_applicants.empty or df_prospects.empty:
    print("\nUm ou mais arquivos não puderam ser carregados.")
else:
    # --- 2. Achatando (Flatten) os Dados Aninhados ---
    print("\n--- Achatando colunas com dados aninhados ---")
    df_jobs = flatten_nested_columns(df_jobs)
    df_applicants = flatten_nested_columns(df_applicants)
    # df_prospects já é "flat"

    # --- 3. Unindo (Merge) os DataFrames com as Chaves Corretas ---
    print("\n--- Unindo os DataFrames ---")

    # Merge 1: Prospects + Applicants
    # Usando 'codigo' de prospects e 'applicant_id' de applicants
    df_merged = pd.merge(
        df_prospects,
        df_applicants,
        left_on='codigo', # Chave correta do exemplo!
        right_on='applicant_id',
        how='inner'
    )
    print(f"Após merge de prospects com applicants, o DataFrame tem {len(df_merged)} linhas.")

    # Merge 2: Resultado + Jobs
    df_final = pd.merge(
        df_merged,
        df_jobs,
        on='vaga_id',
        how='inner',
        suffixes=('_candidato', '_vaga')
    )
    print(f"Após merge final com jobs, o DataFrame tem {len(df_final)} linhas.")

    print("\n\n--- CARGA E MERGE CONCLUÍDOS COM SUCESSO! ---")
    print("\nInformações do DataFrame Final (agora com colunas achatadas):")
    # Usando show_counts=True para ver a contagem de nulos em cada coluna
    df_final.info(verbose=True, show_counts=True)
    
 # Salvando em Parquet para performance e compressão
    try:
        df_final.to_parquet('dados_unificados.parquet', index=False, engine='pyarrow')
        print("\nDataFrame final salvo com sucesso como 'dados_unificados.parquet'")
    except Exception as e:
        print(f"\nNão foi possível salvar o arquivo Parquet. Erro: {e}")