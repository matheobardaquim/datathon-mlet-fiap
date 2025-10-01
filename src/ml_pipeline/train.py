# ml_pipeline/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ml_pipeline import config
from ml_pipeline.data_processing import create_unified_dataset
from ml_pipeline.feature_engineering import create_match_features
from ml_pipeline.evaluate import print_evaluation_metrics

def run_training():
    """Orquestra a pipeline completa de treinamento."""
    print("Iniciando pipeline de treinamento...")
    if not os.path.exists(config.UNIFIED_DATA_PATH):
        print(f"Arquivo '{config.UNIFIED_DATA_PATH}' não encontrado. Executando data processing...")
        create_unified_dataset()
    else:
        print(f"Arquivo '{config.UNIFIED_DATA_PATH}' encontrado.")
    
    df = pd.read_parquet(config.UNIFIED_DATA_PATH)
    df['target'] = df[config.TARGET_COLUMN].apply(lambda x: 1 if x in config.STATUS_POSITIVOS else 0)
    
    df_featured = create_match_features(df)
    
    X = df_featured[config.CATEGORICAL_FEATURES + config.MATCH_FEATURES]
    y = df_featured['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), config.CATEGORICAL_FEATURES)],
        remainder='passthrough'
    )
    model = RandomForestClassifier(**config.MODEL_PARAMS)
    pipeline_final = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    print("Treinando o modelo RandomForest (V2)...")
    pipeline_final.fit(X_train, y_train)
    
    y_pred = pipeline_final.predict(X_test)
    print_evaluation_metrics(y_test, y_pred, model_name="Random Forest Otimizado (V2)")
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print(f"Salvando pipeline final em '{config.MODEL_PATH}'...")
    joblib.dump(pipeline_final, config.MODEL_PATH, compress=True)
    print("Pipeline de treinamento concluída com sucesso!")

if __name__ == "__main__":
    run_training()