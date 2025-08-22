# project_restaurant_revenue

Predicción de **ingresos anuales de restaurantes** (regresión) usando **PyCaret** y despliegue de una app **Streamlit** en **Google Cloud Run**.

La app acepta **CSV RAW** (sin escalar y con categóricas como `City Group` y `Type`), ejecuta un **pipeline de inferencia** y devuelve las predicciones de `revenue`.

---

## 🧭 Contenidos

- [Objetivo y arquitectura](#objetivo-y-arquitectura)
- [Stack y requisitos](#stack-y-requisitos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Artefactos de entrenamiento / inferencia](#artefactos-de-entrenamiento--inferencia)
- [Ejecución local](#ejecución-local)
  - [Con Python/venv](#con-pythonvenv)
  - [Con Docker](#con-docker)
  - [Con Dev Containers (VS Code)](#con-dev-containers-vs-code)
- [Despliegue en Google Cloud Run](#despliegue-en-google-cloud-run)
- [Uso de la app](#uso-de-la-app)
- [Troubleshooting](#troubleshooting)
- [Extras (Hydra, pdoc, pre-commit)](#extras-hydra-pdoc-pre-commit)

---

## 🎯 Objetivo y arquitectura

**Objetivo**: predecir `revenue` de restaurantes a partir de variables como ubicación, tipo y otras características.

**Arquitectura de inferencia (pipeline en producción):**
1. `FeatureDropper` → elimina columnas no necesarias (p. ej. `Id`).
2. `OrdinalEncoderCols` → codifica categóricas (`City Group`, `Type`) con `OrdinalEncoder` (`unknown_value=-1`).
3. `FeatureAligner` → alinea/ordena columnas exactamente como el modelo espera; agrega faltantes con `0.0`.
4. `PrefitMinMaxScaler` → aplica **scaler preentrenado** sobre *selected features*.
5. `PyCaretPredictor` → llama al modelo final guardado por PyCaret y produce la predicción de `revenue`.

La app Streamlit carga un **pipeline ya fiteado** (serializado con joblib), por lo que no reentrena nada en producción.

---

## 🧱 Stack y requisitos

- **Python**: 3.11
- **Librerías principales**:
  - `pycaret==3.3.2`
  - `scikit-learn==1.3.2`
  - `numpy==1.26.4`, `scipy==1.11.4`, `pandas==2.1.4`, `joblib==1.3.2`
  - `lightgbm==4.3.0` (o el backend que use tu modelo final)
  - `streamlit==1.33.0`, `matplotlib==3.7.5`, `seaborn==0.13.2`
- **SO deps**: `libgomp1` (requerido por LightGBM)
- **GCP**: Cloud Build, Artifact Registry, Cloud Run

Archivo de producción: `requirements.txt` ya pinneado para evitar conflictos.

---

## 🗂️ Estructura del proyecto

```bash
.
├── .devcontainer/
│   └── devcontainer.json           # Config del contenedor de desarrollo (Dockerfile.dev)
├── src/
│   ├── modelo_ml_streamlit.py      # App Streamlit (inferencia)
│   ├── images/
│   │   └── datapath-logo.png       # (opcional) logo
│   └── models/
│       └── restaurant_revenue_pipeline.pkl  # Pipeline completo fiteado (joblib)
├── data/
│   ├── raw/                        # CSVs crudos (train/test/unseen)
│   └── processed/                  # CSVs procesados (si se usan para validación)
│       ├── selected_features.csv   # Lista de features finales
│       ├── xtrain.csv / xtest.csv  # Features numéricos (opcional)
│       └── ytrain.csv / ytest.csv  # Target (opcional)
├── models/
│   ├── restaurant_revenue_pycaret.pkl        # Modelo final guardado por PyCaret
│   └── minmax_scaler_selected.joblib         # Scaler entrenado SOLO en selected_features
├── Dockerfile.prod                 # Imagen de producción (Streamlit)
├── cloudbuild.yaml                 # Build y push a Artifact Registry
├── service.yaml                    # Definición del servicio Cloud Run
├── gcr-service-policy.yaml         # IAM policy (invoker)
├── requirements.txt                # Dependencias de producción
├── README.md
└── notebooks/                      # EDA, training, etc.
```

> Nota: `restaurant_revenue_pipeline.pkl` es el artefacto **que se usa en producción** (contiene el encoder, aligner, scaler y el predictor).

---

## 📦 Artefactos de entrenamiento / inferencia

Generados y versionados en este proyecto:

- `models/restaurant_revenue_pycaret.pkl`  
  Modelo final guardado por PyCaret (incluye preprocesado de PyCaret si se usó).
- `models/minmax_scaler_selected.joblib`  
  Scaler **entrenado solo** sobre las `selected_features` (evita mismatch de columnas).
- `data/processed/selected_features.csv`  
  Orden de columnas esperado por el pipeline.
- `src/models/restaurant_revenue_pipeline.pkl`  
  **Pipeline completo de inferencia** (el que carga la app Streamlit).

---

## ▶️ Ejecución local

### Con Python/venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# Ejecutar SIEMPRE desde la raíz del repo
streamlit run src/modelo_ml_streamlit.py
```

> La app busca el pipeline relativo a `src/`, así que ejecuta desde la **raíz** del proyecto.

### Con Docker

```bash
docker build -t restaurant-revenue:prod -f Dockerfile.prod .
docker run --rm -p 8080:8080 -e PORT=8080 restaurant-revenue:prod
# abre http://localhost:8080
```

### Con Dev Containers (VS Code)

- `.devcontainer/devcontainer.json` + `Dockerfile.dev` (Python 3.11-slim + gcloud CLI)  
- Abre con la extensión **Dev Containers** → *Reopen in Container*.

---

## ☁️ Despliegue en Google Cloud Run

1. **Artifact Registry** (una sola vez por repo)
   ```bash
   gcloud artifacts repositories create repo-proyecto-revenue-streamlit \
     --repository-format docker \
     --project mlops12-carlos-sanchez \
     --location us-central1
   ```

2. **Build & Push** con Cloud Build
   ```yaml
   # cloudbuild.yaml
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-f', 'Dockerfile.prod',
              '-t', 'us-central1-docker.pkg.dev/mlops12-carlos-sanchez/repo-proyecto-revenue-streamlit/image-v7-api-datapath:latest', '.']
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', 'us-central1-docker.pkg.dev/mlops12-carlos-sanchez/repo-proyecto-revenue-streamlit/image-v7-api-datapath:latest']
   ```
   ```bash
   gcloud builds submit --config=cloudbuild.yaml --project mlops12-carlos-sanchez
   ```

3. **Cloud Run** (Service manifest)
   ```yaml
   # service.yaml
   apiVersion: serving.knative.dev/v1
   kind: Service
   metadata:
     name: servicio-modelo-revenue-carlos-sanchez
   spec:
     template:
       spec:
         containers:
           - image: us-central1-docker.pkg.dev/mlops12-carlos-sanchez/repo-proyecto-revenue-streamlit/image-v7-api-datapath:latest
             ports:
               - containerPort: 8080
   ```
   ```bash
   gcloud run services replace service.yaml \
     --region us-central1 \
     --project mlops12-carlos-sanchez
   ```

4. **IAM** (hacer público el servicio)
   ```yaml
   # gcr-service-policy.yaml
   bindings:
   - members:
     - allUsers
     role: roles/run.invoker
   ```
   ```bash
   gcloud run services set-iam-policy servicio-modelo-revenue-carlos-sanchez \
     gcr-service-policy.yaml \
     --region us-central1 \
     --project mlops12-carlos-sanchez
   ```

URL de ejemplo:
```
https://servicio-modelo-revenue-carlos-sanchez-770575062081.us-central1.run.app
```

---

## 🖥️ Uso de la app

1. Subir **CSV RAW** con columnas como:
   - `Id` *(opcional)*
   - `City Group` *(categórica)*
   - `Type` *(categórica)*
   - `P1...Pn` *(numéricas)*
   - `revenue` *(opcional; si existe, se usan métricas RMSE/MAE/R²)*

2. La app mostrará:
   - Vista previa (muestra recortada para CSVs grandes)
   - Predicciones (columna `revenue_pred`)
   - Si hay `revenue`, métricas y gráfico “Predicho vs Real”
   - Botón para descargar CSV con predicciones

---

## 🧪 Troubleshooting

- **“No se encontró el pipeline en 'src/models/restaurant_revenue_pipeline.pkl'”**  
  Ejecuta `streamlit run src/modelo_ml_streamlit.py` desde la **raíz** del repo (o usa rutas basadas en `__file__`).  
  Verifica que el `.pkl` esté dentro de la imagen en `src/models/`.

- **`Error cargando pipeline: Can't get attribute 'PyCaretPredictor' ...`**  
  Las clases custom (p. ej. `PyCaretPredictor`, `FeatureDropper`, etc.) deben estar **definidas en el script** donde se hace `joblib.load`, o en un módulo importable estable. Ya están incluidas en `modelo_ml_streamlit.py`.

- **`'numpy.ndarray' object has no attribute 'to_numpy'`**  
  No encadenes `.to_numpy()` sobre algo que **ya** es `ndarray`. Primero `pd.to_numeric(serie).to_numpy()` y después recorta/usa como `ndarray`.

- **`Invalid value for the index parameter. There are duplicate indices ...` (PyCaret)**  
  Resetea el índice del DataFrame de predicción antes de llamar a `predict_model` (el pipeline ya lo hace internamente).

- **`Columnas esperadas por el scaler ausentes`**  
  Asegúrate de usar el **scaler entrenado solo con `selected_features`** (`minmax_scaler_selected.joblib`) y de alinear columnas con `FeatureAligner`.

- **Front-end Streamlit “Importing a module script failed.”**  
  Suele ocurrir por tablas muy grandes (p. ej., 100k filas). Muestra **muestras** (`head(200)`) y ofrece la descarga completa.

- **Cloud Run no responde**  
  - Ver logs:
    ```bash
    gcloud logs tail --region us-central1 --project mlops12-carlos-sanchez \
      --service servicio-modelo-revenue-carlos-sanchez
    ```
  - Asegura `--server.port=${PORT}` en el ENTRYPOINT.  
  - Ajusta memoria/CPU si procesas archivos grandes.

---

## 📎 Extras (Hydra, pdoc, pre-commit)

Este repo viene del template e incluye integración lista para:

- **Hydra**: gestión de configs (opcional para este proyecto)
- **pdoc**: generar documentación de la API
- **pre-commit**: hooks (`ruff`, `black`, `mypy`)

### pre-commit
```bash
pre-commit install
```

### pdoc
```bash
pdoc src -o docs                     # generar estático
pdoc src --http localhost:8080       # server local
```
