# passo_2_e_3_avancado.py (com caminho corrigido)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Carregando os dados unificados...")
# CORREÇÃO: Adicionamos 'data/' para encontrar o arquivo a partir da raiz do projeto.
df = pd.read_parquet('dados_unificados.parquet')

# --- 1. DEFINIÇÃO DA VARIÁVEL ALVO (TARGET) ---
status_positivos = [
    'Contratado pela Decision', 'Contratado como Hunting', 'Encaminhado ao Requisitante',
    'Entrevista Técnica', 'Entrevista com Cliente', 'Aprovado', 'Em avaliação pelo RH',
    'Documentação PJ', 'Documentação CLT', 'Documentação Cooperado',
    'Encaminhar Proposta', 'Proposta Aceita'
]
df['target'] = df['situacao_candidado'].apply(lambda x: 1 if x in status_positivos else 0)

# --- 2. SELEÇÃO DE FEATURES (VERSÃO EXPANDIDA) ---
features_base = [
    'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel profissional',
    'perfil_vaga.nivel_academico', 'perfil_vaga.nivel_ingles'
]
features_novas = [
    'informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao',
    'informacoes_basicas.tipo_contratacao'
]
colunas_selecionadas = features_base + features_novas
df_features = df[colunas_selecionadas + ['target']].copy()

# --- 3. ENGENHARIA DE FEATURES E PRÉ-PROCESSAMENTO ---
print("Iniciando Engenharia de Features e Pré-processamento com mais colunas...")
for col in colunas_selecionadas:
    df_features[col] = df_features[col].fillna('N/A').str.lower()

df_features['match_nivel_profissional'] = np.where(df_features[features_base[0]] == df_features[features_base[3]], 1, 0)
df_features['match_nivel_academico'] = np.where(df_features[features_base[1]] == df_features[features_base[4]], 1, 0)
df_features['match_nivel_ingles'] = np.where(df_features[features_base[2]] == df_features[features_base[5]], 1, 0)
df_features['match_area_atuacao'] = np.where(df_features[features_novas[0]] == df_features[features_novas[1]], 1, 0)

colunas_categoricas = colunas_selecionadas
preprocessor_v2 = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas)],
    remainder='passthrough'
)

X = df_features.drop('target', axis=1)
y = df_features['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. TREINAMENTO DO MODELO (VERSÃO OTIMIZADA) ---
print("\n--- Treinando Modelo Random Forest Otimizado ---")
random_forest_v2 = RandomForestClassifier(
    n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1
)
pipeline_final = Pipeline(steps=[('preprocessor', preprocessor_v2), ('classifier', random_forest_v2)])
pipeline_final.fit(X_train, y_train)

# --- 5. AVALIAÇÃO DO NOVO MODELO ---
print("\n--- Resultados do Random Forest Otimizado (V2) ---")
y_pred_v2 = pipeline_final.predict(X_test)
print("\nRelatório de Classificação (V2):")
print(classification_report(y_test, y_pred_v2))
print("\nMatriz de Confusão (V2):")
print(confusion_matrix(y_test, y_pred_v2))

# --- 6. SALVANDO OS NOVOS ARTEFATOS ---
print("\n--- Salvando o modelo V2 e o preprocessor V2 ---")
joblib.dump(pipeline_final, 'pipeline_model_v2.joblib.gz', compress=True)
print("Pipeline salva como 'pipeline_model_v2.joblib'.")