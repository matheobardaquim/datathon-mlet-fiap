# passo2_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("Carregando os dados unificados...")
df = pd.read_parquet('dados_unificados.parquet')
print("Dados carregados com sucesso.")

# --- A. DEFINIÇÃO DA VARIÁVEL ALVO (TARGET) ---
# Nossa variável mais importante é a 'situacao_candidado'.
# Vamos ver quais são os valores possíveis e suas contagens.
print("\n--- Análise da Variável Alvo: 'situacao_candidado' ---")
print(df['situacao_candidado'].value_counts())

# PARA NOSSA PRIMEIRA VERSÃO, VAMOS CRIAR UMA REGRA SIMPLES:
# Se a situação indicar que o candidato avançou no processo (ex: Contratado, Encaminhado), consideramos 'match' (1).
# Caso contrário, consideramos 'não-match' (0).

# **AÇÃO NECESSÁRIA:** Analise o output acima e defina quais status são positivos.
# Edite a lista 'status_positivos' abaixo com base na sua análise.
status_positivos = [
    'Contratado pela Decision',
    'Contratado como Hunting',
    'Encaminhado ao Requisitante',
    'Entrevista Técnica',
    'Entrevista com Cliente',
    'Aprovado',
    'Em avaliação pelo RH',
    'Documentação PJ',
    'Documentação CLT',
    'Documentação Cooperado',
    'Encaminhar Proposta',
    'Proposta Aceita'
]

df['target'] = df['situacao_candidado'].apply(lambda x: 1 if x in status_positivos else 0)
print(f"\nVariável 'target' criada. Proporção de matches (1): {df['target'].mean():.2%}")


# --- B. SELEÇÃO DE FEATURES ---
# Começaremos com um conjunto simples de features que comparam candidato e vaga.
# Usaremos os níveis de senioridade, acadêmico e de idiomas.

features_candidato = [
    'informacoes_profissionais.nivel_profissional',
    'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles',
]

features_vaga = [
    'perfil_vaga.nivel profissional', # Note o espaço no nome da coluna
    'perfil_vaga.nivel_academico',
    'perfil_vaga.nivel_ingles',
]

# Colunas de texto para uma futura análise com NLP (não usaremos agora)
# features_texto = ['cv_pt', 'perfil_vaga.principais_atividades', 'perfil_vaga.competencia_tecnicas_e_comportamentais']

colunas_selecionadas = features_candidato + features_vaga
df_features = df[colunas_selecionadas + ['target']].copy()


# --- C. ENGENHARIA DE FEATURES E PRÉ-PROCESSAMENTO ---
# Vamos criar features que comparam diretamente o perfil do candidato com o da vaga.
print("\n--- Iniciando Engenharia de Features e Pré-processamento ---")

# 1. Limpeza e Normalização Básica:
#    - Preencher valores nulos com 'N/A'
#    - Converter tudo para minúsculas para padronizar
for col in colunas_selecionadas:
    df_features[col] = df_features[col].fillna('N/A').str.lower()

# 2. Engenharia de Features Simples: Comparação Direta
#    Cria uma feature binária (0 ou 1) se o nível do candidato é o mesmo exigido pela vaga.
df_features['match_nivel_profissional'] = np.where(df_features[features_candidato[0]] == df_features[features_vaga[0]], 1, 0)
df_features['match_nivel_academico'] = np.where(df_features[features_candidato[1]] == df_features[features_vaga[1]], 1, 0)
df_features['match_nivel_ingles'] = np.where(df_features[features_candidato[2]] == df_features[features_vaga[2]], 1, 0)

# 3. Pré-processamento com Scikit-learn Pipeline
#    Vamos transformar as colunas categóricas originais em numéricas usando OneHotEncoder.
#    Isso cria novas colunas para cada categoria (ex: nivel_profissional_senior, nivel_profissional_pleno).

# Definir quais colunas são categóricas para o encoder
colunas_categoricas = colunas_selecionadas

# Criar o transformador que aplica o OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas)
    ],
    remainder='passthrough' # Mantém as outras colunas (nossas features 'match_*')
)

# Separar features (X) e alvo (y)
X = df_features.drop('target', axis=1)
y = df_features['target']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nDados divididos em treino ({len(X_train)} linhas) e teste ({len(X_test)} linhas).")

# Aplicar o pré-processamento
# IMPORTANTE: O 'fit' é feito SOMENTE nos dados de treino para evitar data leakage.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Pré-processamento concluído.")
print("Shape dos dados de treino processados:", X_train_processed.shape)


# --- D. SALVANDO OS RESULTADOS ---
# Salvar os dados processados e o objeto 'preprocessor' para a próxima etapa.
print("\n--- Salvando os artefatos ---")
joblib.dump(preprocessor, 'preprocessor.joblib')
np.save('X_train_processed.npy', X_train_processed)
np.save('X_test_processed.npy', X_test_processed)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)
print("Arquivos salvos: preprocessor.joblib, X_train_processed.npy, X_test_processed.npy, y_train.csv, y_test.csv")