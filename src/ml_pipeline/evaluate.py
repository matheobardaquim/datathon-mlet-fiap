from sklearn.metrics import classification_report, confusion_matrix

def print_evaluation_metrics(y_true, y_pred, model_name="Modelo"):
    """Imprime as métricas de avaliação."""
    print(f"\n--- Resultados de Avaliação para {model_name} ---")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))