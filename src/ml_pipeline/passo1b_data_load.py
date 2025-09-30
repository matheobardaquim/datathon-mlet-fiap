# passo1b_analise_schema.py

import pandas as pd

try:
    print("Lendo o arquivo Parquet...")
    df = pd.read_parquet('dados_unificados.parquet')
    print("Arquivo lido com sucesso.")

    # Abre um arquivo de texto para salvar as informações do schema
    with open('schema.txt', 'w', encoding='utf-8') as f:
        f.write("--- COLUNAS DO DATAFRAME ---\n")
        for col in df.columns:
            f.write(f"{col}\n")
        
        f.write("\n\n--- TIPOS DE DADOS (dtypes) ---\n")
        f.write(str(df.dtypes))

    print("\nAs informações das colunas e tipos de dados foram salvas em 'schema.txt'.")
    print("Por favor, copie o conteúdo desse arquivo e cole na nossa conversa.")

except FileNotFoundError:
    print("ERRO: O arquivo 'dados_unificados.parquet' não foi encontrado.")
    print("Certifique-se de que você executou o script 'passo1_data_load_v4.py' primeiro.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")