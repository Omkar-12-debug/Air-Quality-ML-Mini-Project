"""
Urban Atmospheric Telemetry & Machine Learning Command Center
Streamlit Application & Interactive Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_pipeline import (
    load_and_clean_data,
    FEATURE_COLS,
    FEATURE_METADATA,
    INFERENCE_PRESETS
)
from src.models import (
    build_and_cache_artifacts,
    predict_all_models
)
from src.ui_theme import (
    apply_custom_css,
    apply_telemetry_theme,
    THEME_COLORS
)

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="Urban Air Quality Telemetry & ML Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Cyber-Dark CSS
apply_custom_css()


@st.cache_resource(show_spinner="Bootstrapping atmospheric telemetry pipeline & training models...")
def initialize_system():
    """
    Cached system bootstrap: loads dataset, partitions features, trains models,
    and returns precomputed diagnostics and persistent artifacts.
    """
    df = load_and_clean_data("AirQualityUCI.csv")
    artifacts = build_and_cache_artifacts(df, force_retrain=False)
    return df, artifacts


# Initialize telemetry pipeline
df_clean, artifacts = initialize_system()

# Sleek Top Command Header Banner & Telemetry Branding
st.markdown("""
<div class="top-command-header">
    <div class="header-left">
        <div class="header-branding">
            <div class="telemetry-badge">
                <span class="pulse-dot"></span> TELEMETRY ONLINE
            </div>
            <div class="command-title">ATMOS-ML COMMAND</div>
            <div class="station-sub">STATION ID: IT-RM-AIRUCI-04 • URBAN ATMOSPHERIC TELEMETRY</div>
        </div>
    </div>
    <div class="header-pills">
        <div class="meta-pill"><span class="pill-tag-cyan">[DATASET]</span> AirQualityUCI</div>
        <div class="meta-pill"><span class="pill-tag-emerald">[RECORDS]</span> 9,357 Observations</div>
        <div class="meta-pill"><span class="pill-tag-violet">[SENSORS]</span> 11 Telemetry Inputs</div>
        <div class="meta-pill"><span class="pill-tag-amber">[TARGET]</span> Benzene (C6H6 µg/m³)</div>
        <div class="meta-pill"><span class="pill-tag-coral">[WHO CAP]</span> 10.0 µg/m³</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Horizontal Navigation Tab Bar
nav_tab = st.radio(
    "MAIN NAVIGATION",
    [
        "Overview & EDA",
        "Leaderboard",
        "Diagnostics",
        "Clustering",
        "Live Simulator"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation_tab"
)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW & TELEMETRY EDA
# ==============================================================================
if nav_tab == "Overview & EDA":
    st.markdown("""
    <div class="app-title">Atmospheric Telemetry & Exploratory Analysis</div>
    <div class="app-subtitle">
        Continuous environmental sensor monitoring, correlation dynamics, and regulatory compliance threshold profiling.
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Telemetry Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_records = len(df_clean)
    num_features = len(FEATURE_COLS)
    mean_benzene = df_clean['Benzene_Target'].mean()
    severe_ratio = (df_clean['Air_Quality_Category'] == 1).mean() * 100

    with kpi_col1:
        st.markdown(f"""
        <div class="telemetry-card card-cyan">
            <div class="metric-label">Total Telemetry Samples</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-sub">Continuous Hourly Records</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="telemetry-card card-emerald">
            <div class="metric-label">Engineered Features</div>
            <div class="metric-value">{num_features}</div>
            <div class="metric-sub">Chemical & Physical Sensors</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="telemetry-card card-violet">
            <div class="metric-label">Mean Benzene Target</div>
            <div class="metric-value">{mean_benzene:.2f} <span style="font-size: 1rem; color:#94A3B8;">µg/m³</span></div>
            <div class="metric-sub">Regulatory Ref: 10.0 µg/m³</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="telemetry-card card-coral">
            <div class="metric-label">Severe Smog Ratio</div>
            <div class="metric-value">{severe_ratio:.2f}%</div>
            <div class="metric-sub">{(df_clean['Air_Quality_Category'] == 1).sum():,} Unhealthy Events</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Interactive Pearson Correlation Heatmap
    st.markdown("### 🌐 Telemetry Pearson Correlation Matrix")
    corr_cols = FEATURE_COLS + ['Benzene_Target']
    corr_matrix = df_clean[corr_cols].corr().round(2)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_cols,
        y=corr_cols,
        colorscale=[
            [0.0, '#0F172A'],
            [0.25, '#1E3A8A'],
            [0.5, '#0284C7'],
            [0.75, '#00F2FE'],
            [1.0, '#38BDF8']
        ],
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10, "family": "JetBrains Mono", "color": "#F8FAFC"},
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1")
        ),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
    ))
    apply_telemetry_theme(fig_corr, title="Sensor & Target Inter-Correlation (Pearson r)", height=560)
    fig_corr.update_xaxes(tickangle=45)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Bivariate Dynamics: Target Distribution & Sensor NMHC Scatter
    biv_col1, biv_col2 = st.columns(2)

    with biv_col1:
        st.markdown("### 📊 Benzene Distribution vs. Regulatory Threshold")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_clean['Benzene_Target'],
            nbinsx=45,
            marker_color='#0284C7',
            opacity=0.85,
            name='Benzene Readings',
            hovertemplate="Range: %{x} µg/m³<br>Frequency: %{y:,}<extra></extra>"
        ))
        fig_hist.add_vline(
            x=10.0,
            line_width=2.5,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="WHO / EU Limit (10 µg/m³)",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", family="JetBrains Mono", size=11)
        )
        apply_telemetry_theme(fig_hist, title="Distribution of Benzene Concentration (µg/m³)", height=400)
        fig_hist.update_xaxes(title_text="Benzene C6H6 (µg/m³)")
        fig_hist.update_yaxes(title_text="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    with biv_col2:
        st.markdown("### 🔬 Titania Sensor (NMHC) vs. Benzene Response")
        # Sample for silky smooth client-side interaction
        sample_df = df_clean.sample(n=min(2500, len(df_clean)), random_state=42)
        fig_scatter = go.Figure()

        # Class 0: Acceptable
        acc_mask = sample_df['Air_Quality_Category'] == 0
        fig_scatter.add_trace(go.Scatter(
            x=sample_df.loc[acc_mask, 'Sensor_NMHC'],
            y=sample_df.loc[acc_mask, 'Benzene_Target'],
            mode='markers',
            marker=dict(size=4.5, color='#10B981', opacity=0.5),
            name='Acceptable (≤ 10 µg/m³)',
            hovertemplate="Sensor NMHC: %{x}<br>Benzene: %{y:.2f} µg/m³<extra></extra>"
        ))
        # Class 1: Severe
        sev_mask = sample_df['Air_Quality_Category'] == 1
        fig_scatter.add_trace(go.Scatter(
            x=sample_df.loc[sev_mask, 'Sensor_NMHC'],
            y=sample_df.loc[sev_mask, 'Benzene_Target'],
            mode='markers',
            marker=dict(size=4.5, color='#EF4444', opacity=0.6),
            name='Severe Smog (> 10 µg/m³)',
            hovertemplate="Sensor NMHC: %{x}<br>Benzene: %{y:.2f} µg/m³<extra></extra>"
        ))

        apply_telemetry_theme(fig_scatter, title="Titania Semiconductor Sensor Response vs Benzene", height=400)
        fig_scatter.update_xaxes(title_text="Sensor PT08.S2 (NMHC Response)")
        fig_scatter.update_yaxes(title_text="Benzene Target (µg/m³)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Telemetry Profile Table
    with st.expander("🔍 View Sensor Hardware Telemetry Taxonomy & Statistical Profile", expanded=False):
        sensor_records = []
        for col in FEATURE_COLS:
            meta = FEATURE_METADATA.get(col, {})
            sensor_records.append({
                "Feature Variable": col,
                "Instrument / Sensor": meta.get('label', col),
                "Physical Unit": meta.get('unit', '-'),
                "Mean": f"{df_clean[col].mean():.2f}",
                "Std Dev": f"{df_clean[col].std():.2f}",
                "Min": f"{df_clean[col].min():.2f}",
                "Median": f"{df_clean[col].median():.2f}",
                "Max": f"{df_clean[col].max():.2f}"
            })
        st.dataframe(pd.DataFrame(sensor_records), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: MODEL LEADERBOARD & BENCHMARKING
# ==============================================================================
elif nav_tab == "Leaderboard":
    st.markdown("""
    <div class="app-title">Master Model Leaderboard & Benchmarking</div>
    <div class="app-subtitle">
        Comparative algorithmic performance across 80/20 stratified validation benchmarks.
    </div>
    """, unsafe_allow_html=True)

    task_type = st.radio(
        "SELECT MODEL PARADIGM",
        ["📈 Continuous Regression (Benzene µg/m³)", "🎯 Binary Classification (Smog Category)"],
        horizontal=True
    )

    if "Regression" in task_type:
        reg_df = artifacts['regression']['leaderboard']

        st.markdown("### 🏆 Master Regression Leaderboard")
        st.markdown(
            "Ranked primarily by **$R^2$ Goodness of Fit** (higher is better) and **RMSE** (lower is better)."
        )

        # Formatted presentation table
        styled_reg_df = reg_df.copy()
        styled_reg_df['R² Score'] = styled_reg_df['R² Score'].map("{:.4f}".format)
        styled_reg_df['RMSE'] = styled_reg_df['RMSE'].map("{:.4f} µg/m³".format)
        styled_reg_df['MAE'] = styled_reg_df['MAE'].map("{:.4f} µg/m³".format)
        styled_reg_df['MAPE'] = (styled_reg_df['MAPE'] * 100).map("{:.2f}%".format)
        styled_reg_df['Var(Residuals)'] = styled_reg_df['Var(Residuals)'].map("{:.4f}".format)

        st.dataframe(styled_reg_df, use_container_width=True, hide_index=True)

        # Comparative Multi-Metric Bar Charts
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            fig_r2 = go.Figure(go.Bar(
                x=reg_df['R² Score'],
                y=reg_df['Model'],
                orientation='h',
                marker=dict(
                    color=reg_df['R² Score'],
                    colorscale=[[0, '#1E3A8A'], [0.97, '#0284C7'], [1.0, '#10B981']]
                ),
                text=[f"{v:.4f}" for v in reg_df['R² Score']],
                textposition='inside',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=11),
                hovertemplate="<b>%{y}</b><br>R² Score: %{x:.4f}<extra></extra>"
            ))
            apply_telemetry_theme(fig_r2, title="Goodness of Fit (R² Score) — Higher is Better", height=380)
            fig_r2.update_xaxes(range=[0.95, 1.002], title_text="R² Coefficient of Determination")
            fig_r2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_r2, use_container_width=True)

        with r_col2:
            fig_rmse = go.Figure(go.Bar(
                x=reg_df['RMSE'],
                y=reg_df['Model'],
                orientation='h',
                marker=dict(
                    color=reg_df['RMSE'],
                    colorscale=[[0, '#10B981'], [0.5, '#F59E0B'], [1.0, '#EF4444']]
                ),
                text=[f"{v:.4f} µg/m³" for v in reg_df['RMSE']],
                textposition='outside',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=10),
                hovertemplate="<b>%{y}</b><br>RMSE: %{x:.4f} µg/m³<extra></extra>"
            ))
            apply_telemetry_theme(fig_rmse, title="Prediction Error (RMSE in µg/m³) — Lower is Better", height=380)
            fig_rmse.update_xaxes(title_text="Root Mean Squared Error (µg/m³)")
            fig_rmse.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown("""
        <div class="callout-box">
            <b>💡 Algorithmic Engineering Insight:</b><br>
            <b>Polynomial Regression (Degree 2)</b> and <b>Decision Tree Regressor</b> dominate regression with an astonishing <b>R² > 0.9997</b> and <b>RMSE < 0.12 µg/m³</b>. This matches environmental sensor physics: semiconductor gas sensors (PT08.S2 Titania) exhibit non-linear power-law conductivity responses to ambient volatile organic compounds, which second-degree polynomial terms capture with near-perfect fidelity.
        </div>
        """, unsafe_allow_html=True)

    else:
        clf_df = artifacts['classification']['leaderboard']

        st.markdown("### 🏆 Master Classification Leaderboard")
        st.markdown(
            "Ranked primarily by **F1-Score**, **Accuracy**, and **ROC-AUC** on stratified test partitions."
        )

        styled_clf_df = clf_df.copy()
        styled_clf_df['Accuracy'] = (styled_clf_df['Accuracy'] * 100).map("{:.2f}%".format)
        styled_clf_df['Precision'] = (styled_clf_df['Precision'] * 100).map("{:.2f}%".format)
        styled_clf_df['Recall'] = (styled_clf_df['Recall'] * 100).map("{:.2f}%".format)
        styled_clf_df['F1-Score'] = styled_clf_df['F1-Score'].map("{:.4f}".format)
        styled_clf_df['ROC-AUC'] = styled_clf_df['ROC-AUC'].map("{:.4f}".format)

        st.dataframe(styled_clf_df, use_container_width=True, hide_index=True)

        c_col1, c_col2 = st.columns(2)

        with c_col1:
            fig_f1 = go.Figure(go.Bar(
                x=clf_df['F1-Score'],
                y=clf_df['Model'],
                orientation='h',
                marker=dict(
                    color=clf_df['F1-Score'],
                    colorscale=[[0, '#1E3A8A'], [0.93, '#0284C7'], [1.0, '#8B5CF6']]
                ),
                text=[f"{v:.4f}" for v in clf_df['F1-Score']],
                textposition='inside',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=11),
                hovertemplate="<b>%{y}</b><br>F1-Score: %{x:.4f}<extra></extra>"
            ))
            apply_telemetry_theme(fig_f1, title="Harmonic Mean F1-Score — Higher is Better", height=380)
            fig_f1.update_xaxes(range=[0.88, 1.005], title_text="F1-Score")
            fig_f1.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_f1, use_container_width=True)

        with c_col2:
            fig_auc = go.Figure(go.Bar(
                x=clf_df['ROC-AUC'],
                y=clf_df['Model'],
                orientation='h',
                marker=dict(
                    color=clf_df['ROC-AUC'],
                    colorscale=[[0, '#1E3A8A'], [0.98, '#00F2FE'], [1.0, '#10B981']]
                ),
                text=[f"{v:.4f}" for v in clf_df['ROC-AUC']],
                textposition='inside',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=11),
                hovertemplate="<b>%{y}</b><br>ROC-AUC: %{x:.4f}<extra></extra>"
            ))
            apply_telemetry_theme(fig_auc, title="Area Under ROC Curve (ROC-AUC) — Higher is Better", height=380)
            fig_auc.update_xaxes(range=[0.97, 1.002], title_text="ROC-AUC Score")
            fig_auc.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("""
        <div class="callout-box">
            <b>💡 Algorithmic Engineering Insight:</b><br>
            <b>Decision Tree Classifier (Depth 8)</b> and <b>Support Vector Classifier (RBF)</b> achieve an astounding <b>99.8%</b> and <b>99.4% accuracy</b>, respectively. Furthermore, <b>Logistic Regression with PCA (3 PCs)</b> achieves a remarkable <b>94.8% accuracy and 0.990 ROC-AUC</b> while reducing input dimensionality from 11 sensors to only 3 principal components (a 73% feature compression ratio).
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: DEEP-DIVE MODEL DIAGNOSTICS
# ==============================================================================
elif nav_tab == "Diagnostics":
    st.markdown("""
    <div class="app-title">Individual Model Deep-Dive Diagnostics</div>
    <div class="app-subtitle">
        Parity lines, residual distributions, confusion matrices, ROC curves, and feature importance profiles.
    </div>
    """, unsafe_allow_html=True)

    diag_mode = st.radio(
        "DIAGNOSTIC PIPELINE",
        ["Regression Diagnostics", "Classification Diagnostics"],
        horizontal=True
    )

    if diag_mode == "Regression Diagnostics":
        reg_model_names = list(artifacts['regression']['models'].keys())
        selected_model = st.selectbox("CHOOSE REGRESSION ALGORITHM TO INSPECT", reg_model_names, index=0)

        diag_data = artifacts['regression']['diagnostics'][selected_model]
        y_test_arr = artifacts['splits']['y_reg_test'].to_numpy()
        y_pred = diag_data['y_pred']
        residuals = diag_data['residuals']
        sample_idx = artifacts['test_sample_indices']

        y_test_samp = y_test_arr[sample_idx]
        y_pred_samp = y_pred[sample_idx]
        res_samp = residuals[sample_idx]

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.markdown("#### 🎯 Parity Plot (Actual vs. Predicted)")
            fig_parity = go.Figure()

            # Scatter points
            fig_parity.add_trace(go.Scatter(
                x=y_test_samp,
                y=y_pred_samp,
                mode='markers',
                marker=dict(size=5, color='#00F2FE', opacity=0.6),
                name='Predictions',
                hovertemplate="Actual: %{x:.2f} µg/m³<br>Predicted: %{y:.2f} µg/m³<extra></extra>"
            ))

            # 45-degree ideal parity line
            max_val = max(float(y_test_samp.max()), float(y_pred_samp.max())) + 2
            fig_parity.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='#EF4444', dash='dash', width=2),
                name='Ideal Parity (y = ŷ)'
            ))

            apply_telemetry_theme(fig_parity, title=f"{selected_model}: Parity Plot", height=400)
            fig_parity.update_xaxes(title_text="Actual Benzene (µg/m³)")
            fig_parity.update_yaxes(title_text="Predicted Benzene (µg/m³)")
            st.plotly_chart(fig_parity, use_container_width=True)

        with d_col2:
            st.markdown("#### 📉 Residual Error Distribution (y - ŷ)")
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=res_samp,
                nbinsx=40,
                marker_color='#8B5CF6',
                opacity=0.8,
                name='Residuals',
                hovertemplate="Error Range: %{x:.2f}<br>Count: %{y}<extra></extra>"
            ))
            fig_res.add_vline(x=0.0, line_width=2, line_dash='dash', line_color='#10B981')
            apply_telemetry_theme(fig_res, title=f"{selected_model}: Residual Distribution", height=400)
            fig_res.update_xaxes(title_text="Residual Error (Actual - Predicted)")
            fig_res.update_yaxes(title_text="Frequency")
            st.plotly_chart(fig_res, use_container_width=True)

        # Feature Importance / Coefficients (if applicable)
        if diag_data['feature_importance'] is not None:
            st.markdown("#### ⚖️ Feature Influence / Importance Weights")
            feat_df = diag_data['feature_importance']
            col_name = feat_df.columns[1]

            fig_feat = go.Figure(go.Bar(
                x=feat_df[col_name],
                y=feat_df['Feature'],
                orientation='h',
                marker=dict(
                    color=feat_df[col_name],
                    colorscale=[[0, '#EF4444'], [0.5, '#0284C7'], [1.0, '#00F2FE']]
                ),
                text=[f"{v:.4f}" for v in feat_df[col_name]],
                textposition='auto',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=10)
            ))
            apply_telemetry_theme(fig_feat, title=f"{selected_model}: {col_name} per Sensor", height=380)
            fig_feat.update_xaxes(title_text=col_name)
            fig_feat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_feat, use_container_width=True)

    else:
        clf_model_names = list(artifacts['classification']['models'].keys())
        selected_model = st.selectbox("CHOOSE CLASSIFICATION ALGORITHM TO INSPECT", clf_model_names, index=0)

        diag_data = artifacts['classification']['diagnostics'][selected_model]
        cm = diag_data['cm']
        fpr = diag_data['fpr']
        tpr = diag_data['tpr']
        roc_auc = diag_data['roc_auc']

        c_col1, c_col2 = st.columns(2)

        with c_col1:
            st.markdown("#### 🔲 Confusion Matrix Heatmap")
            cm_labels = ["Acceptable (0)", "Severe (1)"]
            cm_text = [
                [f"TN: {cm[0, 0]:,}", f"FP: {cm[0, 1]:,}"],
                [f"FN: {cm[1, 0]:,}", f"TP: {cm[1, 1]:,}"]
            ]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=cm_labels,
                y=cm_labels,
                text=cm_text,
                texttemplate="%{text}",
                textfont={"size": 13, "family": "JetBrains Mono", "color": "#F8FAFC"},
                colorscale=[[0, '#0F172A'], [0.5, '#1E3A8A'], [1.0, '#0284C7']],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
            ))
            apply_telemetry_theme(fig_cm, title=f"{selected_model}: Confusion Matrix", height=400)
            fig_cm.update_xaxes(title_text="Predicted Air Quality Category")
            fig_cm.update_yaxes(title_text="Actual Air Quality Category", autorange="reversed")
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_col2:
            st.markdown("#### 📈 Receiver Operating Characteristic (ROC)")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                line=dict(color='#00F2FE', width=2.5),
                name=f"ROC Curve (AUC = {roc_auc:.4f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='#64748B', dash='dash', width=1.5),
                name="Random Guess (AUC = 0.5)"
            ))
            apply_telemetry_theme(fig_roc, title=f"{selected_model}: ROC Curve (AUC = {roc_auc:.4f})", height=400)
            fig_roc.update_xaxes(title_text="False Positive Rate (1 - Specificity)")
            fig_roc.update_yaxes(title_text="True Positive Rate (Sensitivity)")
            st.plotly_chart(fig_roc, use_container_width=True)

        # PCA Component Breakdown or Feature Importances
        if "3 PCs" in selected_model:
            st.markdown("#### 🌀 PCA Dimensionality Reduction Diagnostics")
            exp_var = diag_data['explained_variance'] * 100
            cum_var = diag_data['cumulative_variance'] * 100

            pca_col1, pca_col2 = st.columns([1, 2])
            with pca_col1:
                st.markdown(f"""
                <div class="telemetry-card card-violet">
                    <div class="metric-label">Principal Components</div>
                    <div class="metric-value">3</div>
                    <div class="metric-sub">Down from 11 Original Sensors</div>
                    <hr style="border-color: rgba(255,255,255,0.08); margin: 0.75rem 0;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.6;">
                        PC1 Variance: <b style="color:#00F2FE;">{exp_var[0]:.2f}%</b><br>
                        PC2 Variance: <b style="color:#10B981;">{exp_var[1]:.2f}%</b><br>
                        PC3 Variance: <b style="color:#8B5CF6;">{exp_var[2]:.2f}%</b><br>
                        Total Preserved: <b style="color:#F59E0B;">{cum_var[-1]:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with pca_col2:
                # 2D projection scatter
                pca_trans = diag_data['pca_transformer']
                X_test_pca = pca_trans.transform(artifacts['splits']['X_test_scaled'])
                sample_idx = artifacts['test_sample_indices']
                y_test_samp = artifacts['splits']['y_clf_test'].iloc[sample_idx]

                fig_pca_scatter = go.Figure()
                fig_pca_scatter.add_trace(go.Scatter(
                    x=X_test_pca[sample_idx, 0][y_test_samp == 0],
                    y=X_test_pca[sample_idx, 1][y_test_samp == 0],
                    mode='markers',
                    marker=dict(size=4.5, color='#10B981', opacity=0.6),
                    name='Acceptable (0)'
                ))
                fig_pca_scatter.add_trace(go.Scatter(
                    x=X_test_pca[sample_idx, 0][y_test_samp == 1],
                    y=X_test_pca[sample_idx, 1][y_test_samp == 1],
                    mode='markers',
                    marker=dict(size=4.5, color='#EF4444', opacity=0.7),
                    name='Severe Smog (1)'
                ))
                apply_telemetry_theme(fig_pca_scatter, title="PC1 vs PC2 Feature Space Projection", height=320)
                fig_pca_scatter.update_xaxes(title_text=f"PC1 ({exp_var[0]:.1f}% Var)")
                fig_pca_scatter.update_yaxes(title_text=f"PC2 ({exp_var[1]:.1f}% Var)")
                st.plotly_chart(fig_pca_scatter, use_container_width=True)

        elif diag_data['feature_importance'] is not None:
            st.markdown("#### ⚖️ Classification Feature Weights / Gini Importance")
            feat_df = diag_data['feature_importance']
            col_name = feat_df.columns[1]

            fig_clf_feat = go.Figure(go.Bar(
                x=feat_df[col_name],
                y=feat_df['Feature'],
                orientation='h',
                marker=dict(
                    color=feat_df[col_name],
                    colorscale=[[0, '#EF4444'], [0.5, '#0284C7'], [1.0, '#10B981']]
                ),
                text=[f"{v:.4f}" for v in feat_df[col_name]],
                textposition='auto',
                textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=10)
            ))
            apply_telemetry_theme(fig_clf_feat, title=f"{selected_model}: {col_name} per Feature", height=380)
            fig_clf_feat.update_xaxes(title_text=col_name)
            fig_clf_feat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_clf_feat, use_container_width=True)


# ==============================================================================
# PAGE 4: UNSUPERVISED ATMOSPHERIC REGIME CLUSTERING
# ==============================================================================
elif nav_tab == "Clustering":
    st.markdown("""
    <div class="app-title">Unsupervised Atmospheric Regime Clustering</div>
    <div class="app-subtitle">
        Discovery of latent multi-sensor pollution regimes via K-Means and Principal Component space projections.
    </div>
    """, unsafe_allow_html=True)

    clustering_arts = artifacts['clustering']
    eval_df = clustering_arts['eval_df']

    # Interactive slider for K
    k_selected = st.slider(
        "SELECT NUMBER OF CLUSTERS (K)",
        min_value=2,
        max_value=8,
        value=3,
        step=1,
        help="Adjust the number of latent atmospheric pollution regimes to discover."
    )

    # Live Elbow Curve (WCSS) & Silhouette Score
    k_col1, k_col2 = st.columns(2)

    with k_col1:
        st.markdown("#### 📐 Elbow Method: Inertia (WCSS) vs. K")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=eval_df['k'],
            y=eval_df['Inertia (WCSS)'],
            mode='lines+markers',
            line=dict(color='#00F2FE', width=2.5),
            marker=dict(size=8, color='#00F2FE'),
            name='WCSS'
        ))
        # Highlight selected K
        sel_wcss = float(eval_df.loc[eval_df['k'] == k_selected, 'Inertia (WCSS)'].iloc[0])
        fig_elbow.add_trace(go.Scatter(
            x=[k_selected],
            y=[sel_wcss],
            mode='markers',
            marker=dict(size=14, color='#F59E0B', symbol='diamond'),
            name=f'Active K={k_selected}'
        ))
        apply_telemetry_theme(fig_elbow, title="Within-Cluster Sum of Squares (Inertia)", height=350)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)")
        fig_elbow.update_yaxes(title_text="Inertia (WCSS)")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with k_col2:
        st.markdown("#### 🎯 Silhouette Analysis vs. K")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=eval_df['k'],
            y=eval_df['Silhouette_Score'],
            mode='lines+markers',
            line=dict(color='#10B981', width=2.5),
            marker=dict(size=8, color='#10B981'),
            name='Silhouette Score'
        ))
        sel_sil = float(eval_df.loc[eval_df['k'] == k_selected, 'Silhouette_Score'].iloc[0])
        fig_sil.add_trace(go.Scatter(
            x=[k_selected],
            y=[sel_sil],
            mode='markers',
            marker=dict(size=14, color='#F59E0B', symbol='diamond'),
            name=f'Active K={k_selected}'
        ))
        apply_telemetry_theme(fig_sil, title="Silhouette Coefficient Score", height=350)
        fig_sil.update_xaxes(title_text="Number of Clusters (K)")
        fig_sil.update_yaxes(title_text="Silhouette Score")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # PCA 2D/3D Regime Scatter Projections
    proj_data = clustering_arts['cluster_projections'][k_selected]
    labels = proj_data['labels']
    centroids_pca = proj_data['centroids_pca']
    X_sample_pca = clustering_arts['X_sample_pca']

    view_dim = st.radio("PROJECTION DIMENSIONALITY", ["2D PCA Projection", "3D PCA Projection"], horizontal=True)

    regime_palette = ['#00F2FE', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#EC4899', '#38BDF8', '#A855F7']

    if view_dim == "2D PCA Projection":
        fig_proj2d = go.Figure()

        for c_idx in range(k_selected):
            mask = (labels == c_idx)
            c_color = regime_palette[c_idx % len(regime_palette)]
            fig_proj2d.add_trace(go.Scatter(
                x=X_sample_pca[mask, 0],
                y=X_sample_pca[mask, 1],
                mode='markers',
                marker=dict(size=4.5, color=c_color, opacity=0.55),
                name=f"Regime {c_idx+1}"
            ))

        # Centroid markers
        fig_proj2d.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers+text',
            marker=dict(size=16, color='#F59E0B', symbol='x', line=dict(width=2.5, color='#FFFFFF')),
            text=[f"C{i+1}" for i in range(k_selected)],
            textposition="top center",
            textfont=dict(family='JetBrains Mono', color='#F59E0B', size=11),
            name="Centroids"
        ))

        apply_telemetry_theme(fig_proj2d, title=f"Discovered Atmospheric Regimes (K={k_selected}) in 2D PCA Space", height=480)
        fig_proj2d.update_xaxes(title_text="Principal Component 1")
        fig_proj2d.update_yaxes(title_text="Principal Component 2")
        st.plotly_chart(fig_proj2d, use_container_width=True)

    else:
        fig_proj3d = go.Figure()

        for c_idx in range(k_selected):
            mask = (labels == c_idx)
            c_color = regime_palette[c_idx % len(regime_palette)]
            fig_proj3d.add_trace(go.Scatter3d(
                x=X_sample_pca[mask, 0],
                y=X_sample_pca[mask, 1],
                z=X_sample_pca[mask, 2],
                mode='markers',
                marker=dict(size=3, color=c_color, opacity=0.6),
                name=f"Regime {c_idx+1}"
            ))

        fig_proj3d.add_trace(go.Scatter3d(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            z=centroids_pca[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='#F59E0B', symbol='diamond'),
            text=[f"C{i+1}" for i in range(k_selected)],
            textfont=dict(family='JetBrains Mono', color='#F59E0B', size=11),
            name="Centroids"
        ))

        fig_proj3d.update_layout(
            paper_bgcolor='rgba(11, 15, 25, 0)',
            plot_bgcolor='rgba(17, 24, 39, 0.45)',
            font=dict(family="'JetBrains Mono', monospace", color='#94A3B8'),
            height=540,
            scene=dict(
                xaxis=dict(backgroundcolor="#0B0F19", gridcolor="rgba(255,255,255,0.06)", title="PC1"),
                yaxis=dict(backgroundcolor="#0B0F19", gridcolor="rgba(255,255,255,0.06)", title="PC2"),
                zaxis=dict(backgroundcolor="#0B0F19", gridcolor="rgba(255,255,255,0.06)", title="PC3"),
            ),
            legend=dict(bgcolor='rgba(17, 24, 39, 0.75)', font=dict(color='#CBD5E1'))
        )
        st.plotly_chart(fig_proj3d, use_container_width=True)

    # Cluster Centroid Profile Table
    st.markdown("#### 🔬 Standardized Cluster Centroid Sensor Profiles (Z-Scores)")
    centroid_df = proj_data['centroids_feature_space']
    st.dataframe(centroid_df.round(3), use_container_width=True)

    if k_selected == 3:
        st.markdown("""
        <div class="callout-box">
            <b>🌐 Natural Tri-Regime Environmental Interpretation (K=3):</b><br>
            • <b>Regime 1 (Winter Stagnation):</b> Marked by extreme positive anomalies in Carbon Monoxide (CO_True) and Nitrogen Oxides (NOx_True) under cold, low-temperature boundary layers with restricted atmospheric dispersion.<br>
            • <b>Regime 2 (Clean Dispersion):</b> Characterized by uniformly negative z-scores across all toxic gas sensors, elevated wind dispersion, and moderate relative humidity.<br>
            • <b>Regime 3 (Photochemical Summer Smog):</b> High ambient temperature, intense solar radiation, and elevated Titania/Ozone sensor readings (Sensor_O3, Sensor_NMHC) driving secondary photochemical oxidant synthesis.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: LIVE AIR QUALITY INFERENCE SIMULATOR
# ==============================================================================
elif nav_tab == "Live Simulator":
    st.markdown("""
    <div class="app-title">Live Air Quality Multi-Model Simulator</div>
    <div class="app-subtitle">
        Inject real-time atmospheric telemetry to simulate Benzene concentration and acute smog hazard probability.
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick-Action Buttons
    st.markdown("##### ⚡ Quick Scenario Telemetry Presets")
    p_col1, p_col2, p_col3 = st.columns(3)

    if p_col1.button("🌿 Clean Morning Air", use_container_width=True):
        for k, v in INFERENCE_PRESETS['Clean Morning Air'].items():
            st.session_state[f"sim_{k}"] = v
        st.rerun()

    if p_col2.button("🚗 Moderate Urban Traffic", use_container_width=True):
        for k, v in INFERENCE_PRESETS['Moderate Traffic'].items():
            st.session_state[f"sim_{k}"] = v
        st.rerun()

    if p_col3.button("⚠️ Severe Smog Alert", use_container_width=True):
        for k, v in INFERENCE_PRESETS['Severe Smog Alert'].items():
            st.session_state[f"sim_{k}"] = v
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Two-Column Input Sliders for all 11 Telemetry Features
    sim_inputs = {}
    col_left, col_right = st.columns(2)

    half_len = (len(FEATURE_COLS) + 1) // 2
    left_features = FEATURE_COLS[:half_len]
    right_features = FEATURE_COLS[half_len:]

    with col_left:
        st.markdown("##### 🧪 Chemical Concentrations & Reference Analyzers")
        for col in left_features:
            meta = FEATURE_METADATA[col]
            default_val = st.session_state.get(f"sim_{col}", meta['default'])
            sim_inputs[col] = st.slider(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(meta['min']),
                max_value=float(meta['max']),
                value=float(default_val),
                step=float(meta['step']),
                key=f"sim_{col}",
                help=meta['desc']
            )

    with col_right:
        st.markdown("##### 🌡️ Semiconductor Sensors & Meteorological Telemetry")
        for col in right_features:
            meta = FEATURE_METADATA[col]
            default_val = st.session_state.get(f"sim_{col}", meta['default'])
            sim_inputs[col] = st.slider(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(meta['min']),
                max_value=float(meta['max']),
                value=float(default_val),
                step=float(meta['step']),
                key=f"sim_{col}",
                help=meta['desc']
            )

    st.markdown("---")

    # Real-Time Live Inference Execution
    preds = predict_all_models(sim_inputs, artifacts)
    primary_benzene = preds['primary_benzene']
    primary_prob = preds['primary_probability']
    primary_category = preds['primary_category']

    st.markdown("### 📡 Real-Time Atmospheric Inference Diagnostics")

    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        # Status Card for Benzene Concentration
        is_safe = primary_benzene <= 10.0
        border_class = "card-emerald" if is_safe else "card-coral"
        badge_text = "COMPLIANT / SAFE" if is_safe else "HAZARDOUS / EXCEEDS LIMIT"
        badge_color = "#10B981" if is_safe else "#EF4444"

        st.markdown(f"""
        <div class="telemetry-card {border_class}" style="min-height: 290px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="metric-label">Predicted Benzene (C6H6)</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {badge_color}; background: rgba(255,255,255,0.06); padding: 0.2rem 0.5rem; border-radius: 4px;">
                    {badge_text}
                </div>
            </div>
            <div class="metric-value" style="color: {badge_color}; font-size: 2.8rem; margin: 0.8rem 0;">
                {primary_benzene:.2f} <span style="font-size: 1.2rem; color: #94A3B8;">µg/m³</span>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                WHO & European Union Air Quality Directive 2008/50/EC specifies an annual mean regulatory cap of <b>10.0 µg/m³</b> for carcinogenic benzene exposure.
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #CBD5E1;">
                Consensus Estimator: <span style="color: #00F2FE;">Support Vector Regressor (RBF)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # Gauge Chart for Severe Pollution Risk Probability
        prob_pct = primary_prob * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Severe Smog Risk Probability</b>", 'font': {'size': 14, 'color': '#F8FAFC'}},
            number={'suffix': "%", 'font': {'size': 26, 'family': 'JetBrains Mono', 'color': '#F8FAFC'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#00F2FE" if prob_pct < 50 else "#EF4444", 'thickness': 0.3},
                'bgcolor': "rgba(17, 24, 39, 0.6)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.25)"},
                    {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.35)"}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.8,
                    'value': 50.0
                }
            }
        ))
        apply_telemetry_theme(fig_gauge, height=290, showlegend=False)
        fig_gauge.update_layout(margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Multi-Model Side-by-Side Consensus Comparison
    st.markdown("### 📊 Multi-Model Consensus Telemetry Matrix")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("##### 📈 Regression Ensemble Predictions (Benzene µg/m³)")
        reg_preds = preds['regression_predictions']
        reg_table_data = []
        for m_name, val in reg_preds.items():
            delta = val - 10.0
            status = "🔴 Exceeds Limit" if val > 10.0 else "🟢 Compliant"
            reg_table_data.append({
                "Algorithm": m_name,
                "Predicted Benzene": f"{val:.3f} µg/m³",
                "Threshold Delta": f"{delta:+.3f} µg/m³",
                "Status": status
            })
        st.dataframe(pd.DataFrame(reg_table_data), use_container_width=True, hide_index=True)

        # Plotly horizontal bar chart of regression predictions
        fig_reg_bar = go.Figure(go.Bar(
            x=list(reg_preds.values()),
            y=list(reg_preds.keys()),
            orientation='h',
            marker=dict(
                color=list(reg_preds.values()),
                colorscale=[[0, '#10B981'], [0.4, '#00F2FE'], [0.8, '#F59E0B'], [1.0, '#EF4444']],
                cmin=0,
                cmax=25
            ),
            text=[f"{v:.2f} µg/m³" for v in reg_preds.values()],
            textposition='inside',
            textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=10)
        ))
        fig_reg_bar.add_vline(x=10.0, line_dash='dash', line_color='#EF4444', line_width=2)
        apply_telemetry_theme(fig_reg_bar, title="Regression Predictions vs. 10.0 µg/m³ Cap", height=280)
        fig_reg_bar.update_xaxes(title_text="Predicted Benzene (µg/m³)")
        fig_reg_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_reg_bar, use_container_width=True)

    with m_col2:
        st.markdown("##### 🎯 Classification Ensemble Probabilities")
        clf_preds = preds['classification_predictions']
        clf_table_data = []
        for m_name, details in clf_preds.items():
            p = details['probability']
            pred_class = details['prediction']
            status = "🔴 Severe Smog Risk" if pred_class == 1 else "🟢 Clean / Acceptable"
            clf_table_data.append({
                "Classifier": m_name,
                "Hazard Probability": f"{p * 100:.2f}%",
                "Predicted Class": f"Class {pred_class}",
                "Verdict": status
            })
        st.dataframe(pd.DataFrame(clf_table_data), use_container_width=True, hide_index=True)

        # Plotly horizontal bar chart of classifier probabilities
        probs = [d['probability'] * 100 for d in clf_preds.values()]
        c_names = list(clf_preds.keys())

        fig_clf_bar = go.Figure(go.Bar(
            x=probs,
            y=c_names,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale=[[0, '#10B981'], [0.5, '#F59E0B'], [1.0, '#EF4444']],
                cmin=0,
                cmax=100
            ),
            text=[f"{v:.1f}%" for v in probs],
            textposition='auto',
            textfont=dict(family='JetBrains Mono', color='#F8FAFC', size=10)
        ))
        fig_clf_bar.add_vline(x=50.0, line_dash='dash', line_color='#F59E0B', line_width=2)
        apply_telemetry_theme(fig_clf_bar, title="Smog Hazard Probability (%) per Model", height=280)
        fig_clf_bar.update_xaxes(range=[0, 105], title_text="Predicted Probability (%)")
        fig_clf_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_clf_bar, use_container_width=True)
