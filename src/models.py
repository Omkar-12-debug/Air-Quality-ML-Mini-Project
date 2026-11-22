"""
Machine Learning Training, Diagnostics & Inference Engine
Urban Atmospheric Telemetry & Machine Learning Command Center
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    silhouette_score
)

PRIMARY_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
PRIMARY_ARTIFACTS_PATH = os.path.join(PRIMARY_ARTIFACTS_DIR, 'pipeline_artifacts.joblib')
FALLBACK_ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pipeline_artifacts.joblib')


def train_regression_models(
    X_train_s: pd.DataFrame,
    y_train: pd.Series,
    X_test_s: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Trains and benchmarks the 5 regression models specified in the telemetry research:
    1. Linear Regression (OLS)
    2. Polynomial Regression (Degree 2)
    3. K-Nearest Neighbors Regressor (K=7, distance-weighted)
    4. Support Vector Regressor (RBF, C=10.0, gamma=0.05, epsilon=0.1)
    5. Decision Tree Regressor (Depth 8, min_samples_split=5)
    """
    reg_models = {}
    reg_diagnostics = {}
    metrics_list = []

    y_test_arr = y_test.to_numpy()

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    res_lr = y_test_arr - y_pred_lr
    lr_metrics = {
        'Model': 'Linear Regression',
        'R² Score': float(np.round(r2_score(y_test_arr, y_pred_lr), 4)),
        'RMSE': float(np.round(np.sqrt(mean_squared_error(y_test_arr, y_pred_lr)), 4)),
        'MAE': float(np.round(mean_absolute_error(y_test_arr, y_pred_lr), 4)),
        'MAPE': float(np.round(mean_absolute_percentage_error(y_test_arr, y_pred_lr), 4)),
        'Var(Residuals)': float(np.round(np.var(res_lr), 4))
    }
    metrics_list.append(lr_metrics)
    reg_models['Linear Regression'] = lr
    reg_diagnostics['Linear Regression'] = {
        'y_pred': y_pred_lr,
        'residuals': res_lr,
        'feature_importance': pd.DataFrame({
            'Feature': feature_names,
            'Weight': lr.coef_
        }).sort_values(by='Weight', ascending=False)
    }

    # 2. Polynomial Regression (Deg 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)
    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    y_pred_poly = poly_lr.predict(X_test_poly)
    res_poly = y_test_arr - y_pred_poly
    poly_metrics = {
        'Model': 'Polynomial Regression (Deg 2)',
        'R² Score': float(np.round(r2_score(y_test_arr, y_pred_poly), 4)),
        'RMSE': float(np.round(np.sqrt(mean_squared_error(y_test_arr, y_pred_poly)), 4)),
        'MAE': float(np.round(mean_absolute_error(y_test_arr, y_pred_poly), 4)),
        'MAPE': float(np.round(mean_absolute_percentage_error(y_test_arr, y_pred_poly), 4)),
        'Var(Residuals)': float(np.round(np.var(res_poly), 4))
    }
    metrics_list.append(poly_metrics)
    reg_models['Polynomial Regression (Deg 2)'] = (poly, poly_lr)
    reg_diagnostics['Polynomial Regression (Deg 2)'] = {
        'y_pred': y_pred_poly,
        'residuals': res_poly,
        'feature_importance': None
    }

    # 3. KNN Regressor (K=7, distance-weighted)
    knn_reg = KNeighborsRegressor(n_neighbors=7, weights='distance')
    knn_reg.fit(X_train_s, y_train)
    y_pred_knn = knn_reg.predict(X_test_s)
    res_knn = y_test_arr - y_pred_knn
    knn_metrics = {
        'Model': 'KNN Regressor (K=7)',
        'R² Score': float(np.round(r2_score(y_test_arr, y_pred_knn), 4)),
        'RMSE': float(np.round(np.sqrt(mean_squared_error(y_test_arr, y_pred_knn)), 4)),
        'MAE': float(np.round(mean_absolute_error(y_test_arr, y_pred_knn), 4)),
        'MAPE': float(np.round(mean_absolute_percentage_error(y_test_arr, y_pred_knn), 4)),
        'Var(Residuals)': float(np.round(np.var(res_knn), 4))
    }
    metrics_list.append(knn_metrics)
    reg_models['KNN Regressor (K=7)'] = knn_reg
    reg_diagnostics['KNN Regressor (K=7)'] = {
        'y_pred': y_pred_knn,
        'residuals': res_knn,
        'feature_importance': None
    }

    # 4. Support Vector Regressor (SVR)
    svr = SVR(C=10.0, gamma=0.05, epsilon=0.1, kernel='rbf')
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)
    res_svr = y_test_arr - y_pred_svr
    svr_metrics = {
        'Model': 'SVR (RBF Kernel)',
        'R² Score': float(np.round(r2_score(y_test_arr, y_pred_svr), 4)),
        'RMSE': float(np.round(np.sqrt(mean_squared_error(y_test_arr, y_pred_svr)), 4)),
        'MAE': float(np.round(mean_absolute_error(y_test_arr, y_pred_svr), 4)),
        'MAPE': float(np.round(mean_absolute_percentage_error(y_test_arr, y_pred_svr), 4)),
        'Var(Residuals)': float(np.round(np.var(res_svr), 4))
    }
    metrics_list.append(svr_metrics)
    reg_models['SVR (RBF Kernel)'] = svr
    reg_diagnostics['SVR (RBF Kernel)'] = {
        'y_pred': y_pred_svr,
        'residuals': res_svr,
        'feature_importance': None
    }

    # 5. Decision Tree Regressor (Depth 8)
    dt_reg = DecisionTreeRegressor(max_depth=8, min_samples_split=5, random_state=42)
    dt_reg.fit(X_train_s, y_train)
    y_pred_dt = dt_reg.predict(X_test_s)
    res_dt = y_test_arr - y_pred_dt
    dt_metrics = {
        'Model': 'Decision Tree Regressor (Depth 8)',
        'R² Score': float(np.round(r2_score(y_test_arr, y_pred_dt), 4)),
        'RMSE': float(np.round(np.sqrt(mean_squared_error(y_test_arr, y_pred_dt)), 4)),
        'MAE': float(np.round(mean_absolute_error(y_test_arr, y_pred_dt), 4)),
        'MAPE': float(np.round(mean_absolute_percentage_error(y_test_arr, y_pred_dt), 4)),
        'Var(Residuals)': float(np.round(np.var(res_dt), 4))
    }
    metrics_list.append(dt_metrics)
    reg_models['Decision Tree Regressor (Depth 8)'] = dt_reg
    reg_diagnostics['Decision Tree Regressor (Depth 8)'] = {
        'y_pred': y_pred_dt,
        'residuals': res_dt,
        'feature_importance': pd.DataFrame({
            'Feature': feature_names,
            'Gini_Importance': dt_reg.feature_importances_
        }).sort_values(by='Gini_Importance', ascending=False)
    }

    leaderboard = pd.DataFrame(metrics_list).sort_values(
        by=['R² Score', 'RMSE'], ascending=[False, True]
    ).reset_index(drop=True)

    return {
        'models': reg_models,
        'leaderboard': leaderboard,
        'diagnostics': reg_diagnostics
    }


def train_classification_models(
    X_train_s: pd.DataFrame,
    y_train: pd.Series,
    X_test_s: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Trains and benchmarks the 6 classification models specified:
    1. Logistic Regression (Raw Features)
    2. Logistic Regression (3 Principal Components)
    3. K-Nearest Neighbors Classifier (K=7, distance-weighted)
    4. Support Vector Classifier (RBF Kernel, C=10.0, gamma=0.05, prob=True)
    5. Gaussian Naive Bayes
    6. Decision Tree Classifier (Depth 8, min_samples_split=5)
    """
    clf_models = {}
    clf_diagnostics = {}
    metrics_list = []

    y_test_arr = y_test.to_numpy()

    # 1. Logistic Regression (Raw Features)
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_s, y_train)
    y_pred_log = logreg.predict(X_test_s)
    y_prob_log = logreg.predict_proba(X_test_s)[:, 1]
    fpr_log, tpr_log, _ = roc_curve(y_test_arr, y_prob_log)
    roc_auc_log = roc_auc_score(y_test_arr, y_prob_log)
    cm_log = confusion_matrix(y_test_arr, y_pred_log)

    metrics_list.append({
        'Model': 'Logistic Regression (Raw)',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_log), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_log), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_log), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_log), 4)),
        'ROC-AUC': float(np.round(roc_auc_log, 4))
    })
    clf_models['Logistic Regression (Raw)'] = logreg
    clf_diagnostics['Logistic Regression (Raw)'] = {
        'y_pred': y_pred_log,
        'y_prob': y_prob_log,
        'cm': cm_log,
        'fpr': fpr_log,
        'tpr': tpr_log,
        'roc_auc': roc_auc_log,
        'feature_importance': pd.DataFrame({
            'Feature': feature_names,
            'Log_Odds_Weight': logreg.coef_[0]
        }).sort_values(by='Log_Odds_Weight', key=abs, ascending=False)
    }

    # 2. Logistic Regression (3 Principal Components)
    pca_3 = PCA(n_components=3, random_state=42)
    X_train_pca = pca_3.fit_transform(X_train_s)
    X_test_pca = pca_3.transform(X_test_s)

    logreg_pca = LogisticRegression(max_iter=1000, random_state=42)
    logreg_pca.fit(X_train_pca, y_train)
    y_pred_pca = logreg_pca.predict(X_test_pca)
    y_prob_pca = logreg_pca.predict_proba(X_test_pca)[:, 1]
    fpr_pca, tpr_pca, _ = roc_curve(y_test_arr, y_prob_pca)
    roc_auc_pca = roc_auc_score(y_test_arr, y_prob_pca)
    cm_pca = confusion_matrix(y_test_arr, y_pred_pca)

    metrics_list.append({
        'Model': 'Logistic Regression (3 PCs)',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_pca), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_pca), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_pca), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_pca), 4)),
        'ROC-AUC': float(np.round(roc_auc_pca, 4))
    })
    clf_models['Logistic Regression (3 PCs)'] = (pca_3, logreg_pca)
    clf_diagnostics['Logistic Regression (3 PCs)'] = {
        'y_pred': y_pred_pca,
        'y_prob': y_prob_pca,
        'cm': cm_pca,
        'fpr': fpr_pca,
        'tpr': tpr_pca,
        'roc_auc': roc_auc_pca,
        'pca_transformer': pca_3,
        'explained_variance': pca_3.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca_3.explained_variance_ratio_),
        'feature_importance': None
    }

    # 3. KNN Classifier (K=7, distance-weighted)
    knn_clf = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn_clf.fit(X_train_s, y_train)
    y_pred_knnc = knn_clf.predict(X_test_s)
    y_prob_knnc = knn_clf.predict_proba(X_test_s)[:, 1]
    fpr_knnc, tpr_knnc, _ = roc_curve(y_test_arr, y_prob_knnc)
    roc_auc_knnc = roc_auc_score(y_test_arr, y_prob_knnc)
    cm_knnc = confusion_matrix(y_test_arr, y_pred_knnc)

    metrics_list.append({
        'Model': 'KNN Classifier (K=7)',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_knnc), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_knnc), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_knnc), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_knnc), 4)),
        'ROC-AUC': float(np.round(roc_auc_knnc, 4))
    })
    clf_models['KNN Classifier (K=7)'] = knn_clf
    clf_diagnostics['KNN Classifier (K=7)'] = {
        'y_pred': y_pred_knnc,
        'y_prob': y_prob_knnc,
        'cm': cm_knnc,
        'fpr': fpr_knnc,
        'tpr': tpr_knnc,
        'roc_auc': roc_auc_knnc,
        'feature_importance': None
    }

    # 4. Support Vector Classifier (SVC)
    svc = SVC(C=10.0, gamma=0.05, kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    y_pred_svc = svc.predict(X_test_s)
    y_prob_svc = svc.predict_proba(X_test_s)[:, 1]
    fpr_svc, tpr_svc, _ = roc_curve(y_test_arr, y_prob_svc)
    roc_auc_svc = roc_auc_score(y_test_arr, y_prob_svc)
    cm_svc = confusion_matrix(y_test_arr, y_pred_svc)

    metrics_list.append({
        'Model': 'SVC (RBF Kernel)',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_svc), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_svc), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_svc), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_svc), 4)),
        'ROC-AUC': float(np.round(roc_auc_svc, 4))
    })
    clf_models['SVC (RBF Kernel)'] = svc
    clf_diagnostics['SVC (RBF Kernel)'] = {
        'y_pred': y_pred_svc,
        'y_prob': y_prob_svc,
        'cm': cm_svc,
        'fpr': fpr_svc,
        'tpr': tpr_svc,
        'roc_auc': roc_auc_svc,
        'feature_importance': None
    }

    # 5. Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train_s, y_train)
    y_pred_gnb = gnb.predict(X_test_s)
    y_prob_gnb = gnb.predict_proba(X_test_s)[:, 1]
    fpr_gnb, tpr_gnb, _ = roc_curve(y_test_arr, y_prob_gnb)
    roc_auc_gnb = roc_auc_score(y_test_arr, y_prob_gnb)
    cm_gnb = confusion_matrix(y_test_arr, y_pred_gnb)

    metrics_list.append({
        'Model': 'Gaussian Naive Bayes',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_gnb), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_gnb), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_gnb), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_gnb), 4)),
        'ROC-AUC': float(np.round(roc_auc_gnb, 4))
    })
    clf_models['Gaussian Naive Bayes'] = gnb
    clf_diagnostics['Gaussian Naive Bayes'] = {
        'y_pred': y_pred_gnb,
        'y_prob': y_prob_gnb,
        'cm': cm_gnb,
        'fpr': fpr_gnb,
        'tpr': tpr_gnb,
        'roc_auc': roc_auc_gnb,
        'feature_importance': None
    }

    # 6. Decision Tree Classifier (Depth 8)
    dt_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42)
    dt_clf.fit(X_train_s, y_train)
    y_pred_dtc = dt_clf.predict(X_test_s)
    y_prob_dtc = dt_clf.predict_proba(X_test_s)[:, 1]
    fpr_dtc, tpr_dtc, _ = roc_curve(y_test_arr, y_prob_dtc)
    roc_auc_dtc = roc_auc_score(y_test_arr, y_prob_dtc)
    cm_dtc = confusion_matrix(y_test_arr, y_pred_dtc)

    metrics_list.append({
        'Model': 'Decision Tree Classifier (Depth 8)',
        'Accuracy': float(np.round(accuracy_score(y_test_arr, y_pred_dtc), 4)),
        'Precision': float(np.round(precision_score(y_test_arr, y_pred_dtc), 4)),
        'Recall': float(np.round(recall_score(y_test_arr, y_pred_dtc), 4)),
        'F1-Score': float(np.round(f1_score(y_test_arr, y_pred_dtc), 4)),
        'ROC-AUC': float(np.round(roc_auc_dtc, 4))
    })
    clf_models['Decision Tree Classifier (Depth 8)'] = dt_clf
    clf_diagnostics['Decision Tree Classifier (Depth 8)'] = {
        'y_pred': y_pred_dtc,
        'y_prob': y_prob_dtc,
        'cm': cm_dtc,
        'fpr': fpr_dtc,
        'tpr': tpr_dtc,
        'roc_auc': roc_auc_dtc,
        'feature_importance': pd.DataFrame({
            'Feature': feature_names,
            'Gini_Importance': dt_clf.feature_importances_
        }).sort_values(by='Gini_Importance', ascending=False)
    }

    leaderboard = pd.DataFrame(metrics_list).sort_values(
        by=['F1-Score', 'Accuracy', 'ROC-AUC'], ascending=[False, False, False]
    ).reset_index(drop=True)

    return {
        'models': clf_models,
        'leaderboard': leaderboard,
        'diagnostics': clf_diagnostics
    }


def compute_clustering_artifacts(
    X_train_s: pd.DataFrame,
    feature_names: List[str],
    k_range: range = range(2, 9)
) -> Dict[str, Any]:
    """
    Computes Elbow (Inertia/WCSS) and Silhouette scores for K in [2, 8],
    trains KMeans models for each K, fits PCA for regime visualization,
    and constructs cluster centroid profiles.
    """
    k_vals = list(k_range)
    wcss = []
    sil_scores = []
    kmeans_models = {}

    for k in k_vals:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X_train_s)
        wcss.append(km.inertia_)
        sil = silhouette_score(X_train_s, labels, sample_size=2000, random_state=42)
        sil_scores.append(sil)
        kmeans_models[k] = km

    clustering_eval_df = pd.DataFrame({
        'k': k_vals,
        'Inertia (WCSS)': np.round(wcss, 2),
        'Silhouette_Score': np.round(sil_scores, 4)
    })

    # Fit PCA (3 components) on scaled training data for 2D/3D visualizations
    pca_vis = PCA(n_components=3, random_state=42)
    # Downsample points for ultra-responsive Plotly interactive rendering
    sample_indices = np.random.RandomState(42).choice(len(X_train_s), size=min(1800, len(X_train_s)), replace=False)
    X_sample_s = X_train_s.iloc[sample_indices]
    X_sample_pca = pca_vis.fit_transform(X_sample_s)

    # Regime annotations for default K=3
    regime_names_k3 = {
        0: 'Regime 1: Winter Stagnation (High CO/NOx)',
        1: 'Regime 2: Clean Dispersion (Low Pollution)',
        2: 'Regime 3: Photochemical Smog (High Ozone/Temp)'
    }

    # Precompute sample labels and centroids for each k
    cluster_projections = {}
    for k in k_vals:
        km = kmeans_models[k]
        labels = km.predict(X_sample_s)
        centroids_df = pd.DataFrame(km.cluster_centers_, columns=feature_names)
        centroids_pca = pca_vis.transform(centroids_df)
        cluster_projections[k] = {
            'labels': labels,
            'centroids_pca': centroids_pca,
            'centroids_feature_space': pd.DataFrame(
                km.cluster_centers_,
                columns=feature_names,
                index=[f"Regime {i+1}" for i in range(k)]
            )
        }

    return {
        'eval_df': clustering_eval_df,
        'models': kmeans_models,
        'pca_vis': pca_vis,
        'X_sample_pca': X_sample_pca,
        'sample_indices': sample_indices,
        'cluster_projections': cluster_projections,
        'regime_names_k3': regime_names_k3
    }


def build_and_cache_artifacts(
    df_clean: pd.DataFrame,
    force_retrain: bool = False,
    target_path: str = None
) -> Dict[str, Any]:
    """
    Coordinates data partitioning, model training, diagnostics generation,
    and joblib caching for sub-second dashboard load.
    Prioritizes loading pre-computed artifacts from artifacts/ or models/.
    """
    save_path = target_path or PRIMARY_ARTIFACTS_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not force_retrain:
        # Check primary artifacts location first, then fallback
        for candidate_path in [save_path, PRIMARY_ARTIFACTS_PATH, FALLBACK_ARTIFACTS_PATH]:
            if os.path.exists(candidate_path):
                try:
                    artifacts = joblib.load(candidate_path)
                    return artifacts
                except Exception:
                    continue  # Try next candidate or retrain

    from src.data_pipeline import split_and_scale_data, FEATURE_COLS

    splits = split_and_scale_data(df_clean)

    reg_results = train_regression_models(
        splits['X_train_scaled'],
        splits['y_reg_train'],
        splits['X_test_scaled'],
        splits['y_reg_test'],
        FEATURE_COLS
    )

    clf_results = train_classification_models(
        splits['X_train_scaled'],
        splits['y_clf_train'],
        splits['X_test_scaled'],
        splits['y_clf_test'],
        FEATURE_COLS
    )

    clustering_results = compute_clustering_artifacts(
        splits['X_train_scaled'],
        FEATURE_COLS
    )

    # Pre-select downsampled test points (e.g., 600 points) for instant UI parity & residual plots
    test_sample_indices = np.random.RandomState(42).choice(
        len(splits['y_reg_test']),
        size=min(600, len(splits['y_reg_test'])),
        replace=False
    )

    artifacts = {
        'splits': splits,
        'regression': reg_results,
        'classification': clf_results,
        'clustering': clustering_results,
        'feature_cols': FEATURE_COLS,
        'test_sample_indices': test_sample_indices
    }

    joblib.dump(artifacts, save_path, compress=3)
    return artifacts


def predict_all_models(
    input_values: Dict[str, float],
    artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes live multi-model inference against single sensor telemetry vector.
    Returns predicted Benzene concentrations from all 5 regressors and
    severe pollution risk probabilities from all 6 classifiers.
    """
    from src.data_pipeline import FEATURE_COLS

    input_df = pd.DataFrame([[input_values[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    # Standardize input values with feature names to preserve schema
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=FEATURE_COLS
    )

    # 1. Regression Predictions
    reg_models = artifacts['regression']['models']
    reg_preds = {}

    # Linear Regression
    reg_preds['Linear Regression'] = float(reg_models['Linear Regression'].predict(input_scaled)[0])

    # Polynomial Regression
    poly, poly_lr = reg_models['Polynomial Regression (Deg 2)']
    input_poly = poly.transform(input_scaled)
    reg_preds['Polynomial Regression (Deg 2)'] = float(poly_lr.predict(input_poly)[0])

    # KNN Regressor
    reg_preds['KNN Regressor (K=7)'] = float(reg_models['KNN Regressor (K=7)'].predict(input_scaled)[0])

    # SVR
    reg_preds['SVR (RBF Kernel)'] = float(reg_models['SVR (RBF Kernel)'].predict(input_scaled)[0])

    # Decision Tree Regressor
    reg_preds['Decision Tree Regressor (Depth 8)'] = float(reg_models['Decision Tree Regressor (Depth 8)'].predict(input_scaled)[0])

    # 2. Classification Predictions & Probabilities
    clf_models = artifacts['classification']['models']
    clf_preds = {}

    # Logistic Regression (Raw)
    lr_raw = clf_models['Logistic Regression (Raw)']
    clf_preds['Logistic Regression (Raw)'] = {
        'prediction': int(lr_raw.predict(input_scaled)[0]),
        'probability': float(lr_raw.predict_proba(input_scaled)[0][1])
    }

    # Logistic Regression (3 PCs)
    pca_3, lr_pca = clf_models['Logistic Regression (3 PCs)']
    input_pca = pca_3.transform(input_scaled)
    clf_preds['Logistic Regression (3 PCs)'] = {
        'prediction': int(lr_pca.predict(input_pca)[0]),
        'probability': float(lr_pca.predict_proba(input_pca)[0][1])
    }

    # KNN Classifier
    knn_c = clf_models['KNN Classifier (K=7)']
    clf_preds['KNN Classifier (K=7)'] = {
        'prediction': int(knn_c.predict(input_scaled)[0]),
        'probability': float(knn_c.predict_proba(input_scaled)[0][1])
    }

    # SVC
    svc = clf_models['SVC (RBF Kernel)']
    clf_preds['SVC (RBF Kernel)'] = {
        'prediction': int(svc.predict(input_scaled)[0]),
        'probability': float(svc.predict_proba(input_scaled)[0][1])
    }

    # Gaussian Naive Bayes
    gnb = clf_models['Gaussian Naive Bayes']
    clf_preds['Gaussian Naive Bayes'] = {
        'prediction': int(gnb.predict(input_scaled)[0]),
        'probability': float(gnb.predict_proba(input_scaled)[0][1])
    }

    # Decision Tree Classifier
    dt_c = clf_models['Decision Tree Classifier (Depth 8)']
    clf_preds['Decision Tree Classifier (Depth 8)'] = {
        'prediction': int(dt_c.predict(input_scaled)[0]),
        'probability': float(dt_c.predict_proba(input_scaled)[0][1])
    }

    # Primary consensus/champion values (SVR for regression, Decision Tree/SVC for classification)
    primary_benzene = reg_preds['SVR (RBF Kernel)']
    primary_prob = clf_preds['SVC (RBF Kernel)']['probability']
    primary_class = 1 if primary_prob >= 0.50 else 0

    return {
        'primary_benzene': primary_benzene,
        'primary_probability': primary_prob,
        'primary_category': primary_class,
        'regression_predictions': reg_preds,
        'classification_predictions': clf_preds
    }
