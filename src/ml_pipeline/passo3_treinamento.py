# passo3_treinamento.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("--- Carregando os dados de treino e teste ---")
X_train = np.load('X_train_processed.npy')
X_test = np.load('X_test_processed.npy')
y_train = pd.read_csv('y_train.csv').values.ravel() # .ravel() para converter para o formato correto
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Dados carregados. Treino: {X_train.shape}, Teste: {X_test.shape}")

# --- Modelo 1: Regressão Logística (Baseline) ---
print("\n--- Treinando Modelo de Regressão Logística ---")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Previsões
y_pred_log_reg = log_reg.predict(X_test)

# Avaliação
print("\n--- Resultados da Regressão Logística ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_log_reg))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_log_reg))


# --- Modelo 2: Random Forest Classifier ---
print("\n\n--- Treinando Modelo Random Forest ---")
# n_jobs=-1 usa todos os processadores disponíveis para acelerar o treino
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
random_forest.fit(X_train, y_train)

# Previsões
y_pred_rf = random_forest.predict(X_test)

# Avaliação
print("\n--- Resultados do Random Forest ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_rf))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_rf))


# --- Salvando o Melhor Modelo ---
# Com base nos resultados, o Random Forest geralmente é superior.
print("\n--- Salvando o modelo Random Forest ---")
joblib.dump(random_forest, 'random_forest_model.joblib')
print("Modelo salvo como 'random_forest_model.joblib'.")