import pandas as pd
import numpy as np
import joblib

def load_pipeline(pipeline_path: str = '../models/restaurant_revenue_pipeline.pkl'):
    """Carga el pipeline completo de revenue (drop→encode→align→scale→predict)."""
    return joblib.load(pipeline_path)

def predict_revenue(csv_file_path: str,
                    pipeline_path: str = '../models/restaurant_revenue_pipeline.pkl',
                    return_dataframe: bool = True):
    """Predice revenue a partir de un CSV con datos RAW (sin escalar).

    - El CSV debe contener las columnas esperadas por el pipeline (p.ej. 'Id', 'City Group', 'Type', 'P1'...'Pn').
    - Si el CSV trae columna 'revenue', se ignora para la predicción (no se usa en inferencia).
    - Devuelve un array con predicciones o un DataFrame con columna 'revenue_pred'.
    """
    pipe = load_pipeline(pipeline_path)
    df = pd.read_csv(csv_file_path)
    X = df.drop(columns=['revenue'], errors='ignore')

    y_pred = pipe.predict(X)

    if return_dataframe:
        out = df.copy()
        out['revenue_pred'] = y_pred
        return out

    return y_pred

def predict_revenue_from_df(df: pd.DataFrame,
                            pipeline_path: str = '../models/restaurant_revenue_pipeline.pkl',
                            return_dataframe: bool = True):
    """Igual que predict_revenue pero recibiendo un DataFrame en memoria."""
    pipe = load_pipeline(pipeline_path)
    X = df.drop(columns=['revenue'], errors='ignore')
    y_pred = pipe.predict(X)

    if return_dataframe:
        out = df.copy()
        out['revenue_pred'] = y_pred
        return out

    return y_pred
