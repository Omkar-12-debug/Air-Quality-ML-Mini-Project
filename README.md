# 📡 Urban Atmospheric Telemetry & Machine Learning Command Center

An interactive, production-grade Machine Learning telemetry dashboard and decision-support web application for urban atmospheric pollution monitoring, predictive modeling, and real-time hazard simulation. Built on the **UCI Air Quality Dataset** (9,357 hourly ground-truth & multi-sensor telemetry records) using **Streamlit**, **Plotly**, and **Scikit-Learn**.

---

## 🌟 Executive Highlights & Design Aesthetic

- **Command Center Aesthetic:** Designed with a sleek, telemetry-inspired dark mode (`#0B0F19` deep space background, `#111827` glassmorphic cards with glowing cyan, emerald, and violet borders, monospace data telemetry typography, and micro-animations).
- **Dual Machine Learning Targets:**
  1. **Continuous Regression:** Hourly Benzene Concentration (`C6H6(GT)` in µg/m³).
  2. **Binary Classification:** Severe Air Pollution Hazard (`1` if Benzene $> 10.0$ µg/m³ according to WHO/EU directives, else `0`).
- **Instantaneous Startup & Caching:** Models and diagnostics are automatically pre-trained and serialized via `joblib` into `models/pipeline_artifacts.joblib`, ensuring sub-second page switching and zero lag during interactive inference.
- **Full-Stack Modular Architecture:** Clean separation of concerns across data engineering (`src/data_pipeline.py`), model training & diagnostics (`src/models.py`), UI design system (`src/ui_theme.py`), and dashboard orchestration (`app.py`).

---

## 🏗️ Project Architecture

```
MINI_PROJECT/
├── AirQualityUCI.csv           # 9,357 hourly sensor records from Italian city station
├── Mini_Project.ipynb          # Original research and prototyping notebook
├── app.py                      # Multi-module Streamlit interactive dashboard
├── requirements.txt            # Pinned production dependencies
├── README.md                   # Complete architectural documentation
├── models/
│   └── pipeline_artifacts.joblib # Persistent model weights, scalers & diagnostics cache
└── src/
    ├── __init__.py
    ├── data_pipeline.py        # Sanitation, interpolation, scaling & feature metadata
    ├── models.py               # 5 Regressors, 6 Classifiers, K-Means & inference engine
    └── ui_theme.py             # Cyber-dark CSS injection & Plotly telemetry layouts
```

---

## 📊 5 Interactive Dashboard Views

### 1. 📡 Executive Overview & Telemetry EDA
- **Real-Time KPI Cards:** Total Telemetry Samples (9,357), Cleaned Sensor Channels (11), Mean Benzene ($10.18$ µg/m³), and Severe Smog Event Ratio ($40.56\%$).
- **Interactive Pearson Correlation Heatmap:** Complete cross-correlation matrix with hover telemetry and gradient mapping.
- **Bivariate Distribution Analysis:** Interactive Benzene distribution with WHO/EU $10.0$ µg/m³ threshold line alongside Titania sensor (`Sensor_NMHC`) response scatter.
- **Hardware Sensor Taxonomy Table:** Inspect physical units, instruments, and statistical distributions for all 11 features.

### 2. 🏆 Model Leaderboard & Benchmarking
- **Dual Paradigm Toggle:** Instant toggle between 5 Continuous Regression models and 6 Binary Classification models.
- **Multi-Metric Comparison Tables:** Sortable metrics ($R^2$, RMSE, MAE, MAPE, Var(Residuals) for regression; Accuracy, Precision, Recall, F1-Score, ROC-AUC for classification).
- **Grouped Visual Bar Charts:** Interactive Plotly charts ranking algorithm efficacy.
- **Engineering Insights:** Polynomial Regression ($R^2 = 0.9998$) and Decision Trees ($R^2 = 0.9997$) capture the semiconductor sensors' non-linear conductivity curves.

### 3. 🔬 Deep-Dive Model Diagnostics
- **Per-Algorithm Inspection:** Select any individual model to inspect validation telemetry:
  - **Regression:** $45^\circ$ Parity Plot (Actual vs. Predicted with residual hover), Residual Error Distribution, and Feature Weights / Gini Importance.
  - **Classification:** Confusion Matrix Heatmap (TN, FP, FN, TP breakdown), ROC-AUC Curves, and PCA 2D/3D component variance projections.

### 4. 🌀 Unsupervised Atmospheric Regime Clustering (K-Means)
- **Dynamic Cluster Slider ($K \in [2, 8]$):** Live updates of Within-Cluster Sum of Squares (Elbow curve) and Silhouette Scores.
- **2D & 3D PCA Regime Visualizations:** Interactive projections showing cluster partitions and centroid coordinates ($C_1 \dots C_k$).
- **Tri-Regime Environmental Interpretation ($K=3$):**
  - **Regime 1 (Winter Stagnation):** Temperature inversions trapping CO and NOx.
  - **Regime 2 (Clean Atmospheric Dispersion):** High atmospheric mixing and low pollutant loads.
  - **Regime 3 (Photochemical Summer Smog):** Solar radiation driving high ozone and photochemical reactivity.

### 5. ⚡ Live Air Quality Simulator
- **Quick Preset Buttons:** One-click simulation for *"Clean Morning Air"*, *"Moderate Urban Traffic"*, and *"Severe Smog Alert"*.
- **11 Parameter Interactive Sliders:** Full control over chemical concentrations and meteorological inputs.
- **Real-Time Hazard Assessment:** Predicted Benzene concentration, compliance badge, and severe pollution risk probability gauge ($0\% - 100\%$).
- **Ensemble Consensus Matrix:** Side-by-side predictions across all 5 regressors and 6 classifiers.

---

## 🔬 Telemetry Sensor & Feature Taxonomy

| Feature Variable | Instrument / Sensor Hardware | Physical Unit | Domain Range |
| :--- | :--- | :--- | :--- |
| `CO_True` | Reference NDIR Carbon Monoxide Analyzer | $\text{mg/m}^3$ | $0.1 - 12.0$ |
| `Sensor_CO` | PT08.S1 Tin Oxide Semiconductor Sensor | $\text{mV/}\Omega$ | $640 - 2,100$ |
| `Sensor_NMHC` | PT08.S2 Titania NMHC / VOC Sensor | $\text{mV/}\Omega$ | $380 - 2,250$ |
| `NOx_True` | Reference Chemiluminescent Analyzer | $\text{ppb}$ | $2.0 - 1,500$ |
| `Sensor_NOx` | PT08.S3 Tungsten Oxide Sensor | $\text{mV/}\Omega$ | $320 - 2,700$ |
| `NO2_True` | Reference Analyzer | $\mu\text{g/m}^3$ | $2.0 - 350$ |
| `Sensor_NO2` | PT08.S4 Tungsten Oxide Sensor | $\text{mV/}\Omega$ | $550 - 2,800$ |
| `Sensor_O3` | PT08.S5 Indium Oxide Ozone Sensor | $\text{mV/}\Omega$ | $220 - 2,550$ |
| `Temperature` | Ambient Station Thermocouple | $^\circ\text{C}$ | $-5.0 - 45.0$ |
| `Relative_Humidity` | Capacitive Humidity Sensor | $\%$ | $5.0 - 90.0$ |
| `Absolute_Humidity` | Calculated Volumetric Moisture | $\text{g/m}^3$ | $0.1 - 2.5$ |

---

## 🚀 Local Installation & Execution

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Clone or Navigate to Directory
```bash
cd "d:/Omkar/VIT Stuff/Program files/ML/MINI_PROJECT"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The command center will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Render Free Tier)

This application is optimized for zero-configuration deployment on **Render**, **Streamlit Community Cloud**, or **Railway**:

1. **Push repository to GitHub.**
2. **On Render (Web Service):**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. The application will build, load cached model artifacts, and expose the live dashboard on Render's public URL.
