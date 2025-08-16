import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import os
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================
# Transformadores (deben existir para cargar el pipeline pickled)
# =========================================

from pycaret.regression import load_model, predict_model

class PyCaretPredictor(BaseEstimator):
    """Paso final que usa el modelo guardado con PyCaret para predecir."""
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None

    def fit(self, X, y=None):
        # Si el pipeline se guardó ya fiteado, al cargar desde joblib
        # 'self.model' ya podría venir serializado. Este check no estorba.
        if self.model is None and self.model_path:
            self.model = load_model(self.model_path)
        return self

    def predict(self, X):
        df = pd.DataFrame(X).reset_index(drop=True)
        out = predict_model(self.model, data=df)
        label_col = "Label" if "Label" in out.columns else (
            "prediction_label" if "prediction_label" in out.columns else None
        )
        if label_col is None:
            raise RuntimeError(f"No se encontró la columna de predicción en {out.columns.tolist()}")
        return out[label_col].to_numpy()

class FeatureDropper(BaseEstimator, TransformerMixin):
    """Elimina columnas no necesarias (e.g., 'Id'). Comparación case-insensitive."""
    def __init__(self, features_to_drop=('Id',), case_insensitive=True):
        self.features_to_drop = list(features_to_drop)
        self.case_insensitive = case_insensitive
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if self.case_insensitive:
            targets = {c.lower() for c in self.features_to_drop}
            to_drop = [c for c in X.columns if c.lower() in targets]
        else:
            to_drop = [c for c in self.features_to_drop if c in X.columns]
        return X.drop(columns=to_drop, errors='ignore')

class OrdinalEncoderCols(BaseEstimator, TransformerMixin):
    """Encoda columnas categóricas específicas con OrdinalEncoder (unknown_value=-1)."""
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols
        self.encoder = None
        self.fitted_cols_ = None
    def fit(self, X, y=None):
        X = X.copy()
        if self.cols is None:
            self.fitted_cols_ = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
        else:
            self.fitted_cols_ = [c for c in self.cols if c in X.columns]
        if len(self.fitted_cols_) == 0:
            self.encoder = None
            return self
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.encoder.fit(X[self.fitted_cols_])
        return self
    def transform(self, X):
        X = X.copy()
        if self.encoder is None or len(self.fitted_cols_) == 0:
            return X
        X.loc[:, self.fitted_cols_] = self.encoder.transform(X[self.fitted_cols_])
        return X

class FeatureAligner(BaseEstimator, TransformerMixin):
    """Alinea columnas al orden esperado; agrega faltantes y resetea índice."""
    def __init__(self, features: List[str], fill_missing: float = 0.0, keep_extra: bool = False):
        self.features = list(features)
        self.fill_missing = fill_missing
        self.keep_extra = keep_extra
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.features:
            if c not in X.columns:
                X[c] = self.fill_missing
        cols = self.features + ([c for c in X.columns if c not in self.features] if self.keep_extra else [])
        return X[cols].reset_index(drop=True)

class PrefitMinMaxScaler(BaseEstimator, TransformerMixin):
    """Aplica un MinMaxScaler ya entrenado (joblib) sobre columnas numéricas."""
    def __init__(self, scaler_path: str, columns: Optional[List[str]] = None):
        self.scaler_path = scaler_path
        self.columns = columns
        self.scaler = None
    def fit(self, X, y=None):
        self.scaler = joblib.load(self.scaler_path)
        if self.columns is None:
            self.columns = list(getattr(self.scaler, "feature_names_in_", [])) \
                           or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        return self
    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.columns if c in X.columns]
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columnas esperadas por el scaler ausentes: {missing}")
        X.loc[:, cols] = self.scaler.transform(X[cols])
        return X

# Nota: el pipeline pickled contiene un estimador final que llama internamente a PyCaret.
# Aquí NO necesitamos re-definir PyCaretPredictor; está dentro del pipeline serializado.


# =========================================
# Configuración Streamlit
# =========================================
st.set_page_config(
    page_title="Restaurant Revenue Predictor",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Restaurant Revenue Predictor")
st.markdown("### Predicción de ingresos anuales de restaurantes con ML")

# Logo opcional
try:
    image = Image.open('src/images/datapath-logo.png')
    st.image(image, use_container_width=True)
except Exception:
    st.info("💡 Coloca un logo en `src/images/datapath-logo.png` (opcional).")

# =========================================
# Carga del pipeline
# =========================================
@st.cache_resource
def cargar_pipeline(path: str = 'src/models/restaurant_revenue_pipeline.pkl'):
    if not os.path.exists(path):
        st.error(f"No se encontró el pipeline en '{path}'. Asegúrate de copiarlo dentro de la imagen.")
        return None
    try:
        pipe = joblib.load(path)
        return pipe
    except Exception as e:
        st.error(f"Error cargando pipeline: {e}")
        return None

pipeline = cargar_pipeline()

# =========================================
# Sidebar: carga de CSV RAW
# =========================================
st.sidebar.header("📁 Cargar datos RAW (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona un archivo CSV",
    type=['csv'],
    help="Debe contener columnas como: Id, City Group, Type, P1...Pn. Si trae 'revenue', se usa solo para métricas."
)

with st.sidebar.expander("📋 Formato sugerido"):
    st.markdown("""
    - **Id** (opcional)
    - **City Group** *(categórica)*  
    - **Type** *(categórica)*  
    - **P1...Pn** *(numéricas)*  
    - **revenue** *(opcional, solo para evaluación)*
    """)

# =========================================
# Predicción
# =========================================
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ Archivo cargado: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")

        st.markdown("#### 👀 Vista previa")
        st.dataframe(df_raw.head(5), use_container_width=True)

        # Botón de predicción
        if st.sidebar.button("🚀 Predecir ingresos", type="primary", use_container_width=True):
            if pipeline is None:
                st.error("❌ No se pudo cargar el pipeline.")
            else:
                with st.spinner("🔄 Procesando y prediciendo..."):
                    # Hacer predicciones (pipeline RAW: drop → encode → align → scale → predict)
                    X = df_raw.drop(columns=['revenue'], errors='ignore')
                    y_true_series = df_raw['revenue'] if 'revenue' in df_raw.columns else None

                    y_pred = pipeline.predict(X)

                st.success("✅ ¡Predicciones listas!")

                # Resultados
                resultados = df_raw.copy()
                resultados['revenue_pred'] = y_pred

                # Si hay ground-truth, mostrar métricas
                if y_true_series is not None:
                    # Asegurar Serie → numérico → ndarray sin encadenar .to_numpy() sobre un ndarray
                    y_true_np = pd.to_numeric(y_true_series, errors='coerce').to_numpy()
                    n = min(len(y_pred), len(y_true_np))
                    y_true_np = y_true_np[:n]
                    y_pred_np = y_pred[:n]

                    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
                    mae  = float(mean_absolute_error(y_true_np, y_pred_np))
                    r2   = float(r2_score(y_true_np, y_pred_np))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{rmse:,.2f}")
                    col2.metric("MAE",  f"{mae:,.2f}")
                    col3.metric("R²",   f"{r2:.3f}")

                    # Scatter Predicho vs Real
                    st.markdown("#### 📈 Predicho vs Real")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_true_np, y_pred_np, alpha=0.6)
                    lims = [min(y_true_np.min(), y_pred_np.min()), max(y_true_np.max(), y_pred_np.max())]
                    ax.plot(lims, lims, '--', alpha=0.7)
                    ax.set_xlabel("Real (revenue)")
                    ax.set_ylabel("Predicho (revenue)")
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                # Distribución de predicciones
                st.markdown("#### 📊 Distribución de ingresos predichos")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.hist(resultados['revenue_pred'].values, bins=30, edgecolor='black', alpha=0.7)
                ax2.set_xlabel("Revenue predicho")
                ax2.set_ylabel("Frecuencia")
                ax2.grid(alpha=0.2)
                st.pyplot(fig2)

                # Tabla y descarga (muestra solo una parte para no saturar el frontend)
                st.markdown("#### 📋 Resultados (muestra)")
                st.dataframe(resultados.head(50), use_container_width=True)  # 👈 limita filas para evitar el error del frontend
                csv = resultados.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Descargar CSV de predicciones",
                    data=csv,
                    file_name="restaurant_revenue_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error procesando el CSV: {e}")
else:
    st.info("Carga un CSV en la barra lateral para comenzar.")

# Footer
st.markdown("---")
st.markdown("🍽️ **Restaurant Revenue Predictor** — Streamlit + Scikit-learn + PyCaret")
