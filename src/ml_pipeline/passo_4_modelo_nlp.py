# passo_4_modelo_nlp.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Carregando os dados unificados...")
df = pd.read_parquet('dados_unificados.parquet')

# --- 1. LIMPEZA DOS DADOS DE TEXTO ---
# Para NLP, é crucial limpar os textos e preencher valores nulos
df['cv_pt_clean'] = df['cv_pt'].fillna('').str.lower()
df['competencias_vaga_clean'] = df['perfil_vaga.competencia_tecnicas_e_comportamentais'].fillna('').str.lower()


# --- 2. ENGENHARIA DE FEATURE COM NLP (SIMILARIDADE) ---
print("Iniciando engenharia de features com NLP. Isso pode demorar um pouco...")

# Criar um vocabulário único a partir de todos os textos (CVs e Vagas)
corpus = pd.concat([df['cv_pt_clean'], df['competencias_vaga_clean']], ignore_index=True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500) # Usamos 'english' pois muitos termos de TI são em inglês
tfidf_vectorizer.fit(corpus)

# Transformar os textos em vetores TF-IDF
vectors_cv = tfidf_vectorizer.transform(df['cv_pt_clean'])
vectors_vaga = tfidf_vectorizer.transform(df['competencias_vaga_clean'])

# Calcular a similaridade do cosseno para cada par (CV, Vaga)
# O resultado é uma pontuação de similaridade para cada linha do DataFrame
similarities = [cosine_similarity(vectors_cv[i], vectors_vaga[i])[0][0] for i in range(vectors_cv.shape[0])]
df['similaridade_texto'] = similarities
print("Feature 'similaridade_texto' criada com sucesso.")


# --- 3. PREPARANDO DADOS PARA O MODELO FINAL ---
# Vamos usar a nova feature de similaridade junto com as features do modelo V2
df_features_v2 = pd.read_parquet('dados_unificados.parquet') # Recarrega para pegar as colunas originais
features_base = [
    'informacoes_profissionais.nivel_profissional', 'formacao_e_idiomas.nivel_academico',
    'formacao_e_idiomas.nivel_ingles', 'perfil_vaga.nivel profissional',
    'perfil_vaga.nivel_academico', 'perfil_vaga.nivel_ingles'
]
features_novas = [
    'informacoes_profissionais.area_atuacao', 'perfil_vaga.areas_atuacao',
    'informacoes_basicas.tipo_contratacao'
]
colunas_categoricas = features_base + features_novas
for col in colunas_categoricas:
    df_features_v2[col] = df_features_v2[col].fillna('N/A').str.lower()

X_categorico = pd.get_dummies(df_features_v2[colunas_categoricas], drop_first=True)
X_final = pd.concat([X_categorico.reset_index(drop=True), df[['similaridade_texto']].reset_index(drop=True)], axis=1)

# Definindo o target
status_positivos = [
    'Contratado pela Decision', 'Contratado como Hunting', 'Encaminhado ao Requisitante',
    'Entrevista Técnica', 'Entrevista com Cliente', 'Aprovado', 'Em avaliação pelo RH',
    'Documentação PJ', 'Documentação CLT', 'Documentação Cooperado',
    'Encaminhar Proposta', 'Proposta Aceita'
]
y = df['situacao_candidado'].apply(lambda x: 1 if x in status_positivos else 0)


# --- 4. TREINAMENTO E AVALIAÇÃO DO MODELO V3 ---
print("\n--- Treinando Modelo Final (V3 com NLP) ---")
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

model_v3 = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
model_v3.fit(X_train, y_train)

print("\n--- Resultados do Modelo Final (V3) ---")
y_pred_v3 = model_v3.predict(X_test)
print("\nRelatório de Classificação (V3):")
print(classification_report(y_test, y_pred_v3))
print("\nMatriz de Confusão (V3):")
print(confusion_matrix(y_test, y_pred_v3))

# (Opcional) Salvar o modelo final. Para usar na API, teríamos que refatorar
# o código de pré-processamento em uma pipeline, como fizemos antes.
# Por enquanto, vamos focar na melhoria de performance.