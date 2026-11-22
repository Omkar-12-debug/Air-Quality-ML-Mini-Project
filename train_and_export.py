"""
Standalone Model Training & Artifact Export Pipeline
Pre-trains all models and exports lightweight serialized artifacts to artifacts/
for sub-2-second cold starts on Render's 512 MB free tier.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np

from src.data_pipeline import load_and_clean_data, FEATURE_COLS
from src.models import build_and_cache_artifacts


def main():
    print("=" * 70)
    print("🚀 ATMOSPHERIC TELEMETRY ML: EXPORT & COLD-START OPTIMIZATION")
    print("=" * 70)
    t0 = time.time()

    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    target_joblib = os.path.join(artifacts_dir, 'pipeline_artifacts.joblib')
    target_json = os.path.join(artifacts_dir, 'metrics_summary.json')

    # 1. Ingest & sanitize data
    print("\n[1/4] Ingesting and sanitizing AirQualityUCI.csv...")
    csv_path = os.path.join(os.path.dirname(__file__), 'AirQualityUCI.csv')
    df_clean = load_and_clean_data(csv_path)
    print(f"      ✓ Records: {len(df_clean):,} | Features: {len(FEATURE_COLS)}")

    # 2. Train models and generate diagnostics
    print("\n[2/4] Training 5 Regressors, 6 Classifiers & K-Means Clusters...")
    artifacts = build_and_cache_artifacts(df_clean, force_retrain=True, target_path=target_joblib)
    print("      ✓ All models trained & diagnostic curves computed successfully.")

    # 3. Export metrics summary JSON
    print("\n[3/4] Exporting lightweight metrics summary JSON...")
    reg_leaderboard = artifacts['regression']['leaderboard'].to_dict(orient='records')
    clf_leaderboard = artifacts['classification']['leaderboard'].to_dict(orient='records')
    clustering_eval = artifacts['clustering']['eval_df'].to_dict(orient='records')

    summary_payload = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_observations': len(df_clean),
            'features': FEATURE_COLS,
            'stratified_split': '80/20 (random_state=42)',
            'scaler': 'StandardScaler',
            'regulatory_cap_benzene_ug_m3': 10.0
        },
        'regression_leaderboard': reg_leaderboard,
        'classification_leaderboard': clf_leaderboard,
        'clustering_evaluation': clustering_eval
    }

    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2)
    print(f"      ✓ Metrics JSON exported to: {target_json}")

    # 4. Verification and Cold-Start benchmark
    print("\n[4/4] Verifying artifact integrity & cold-start load time...")
    t_load_start = time.time()
    loaded_artifacts = joblib.load(target_joblib)
    load_duration = time.time() - t_load_start

    joblib_size_mb = os.path.getsize(target_joblib) / (1024 * 1024)
    json_size_kb = os.path.getsize(target_json) / 1024

    print(f"      ✓ Artifact Size (Compressed Joblib): {joblib_size_mb:.2f} MB")
    print(f"      ✓ Metrics JSON Size: {json_size_kb:.1f} KB")
    print(f"      ✓ Cold-Start Load Time: {load_duration:.3f} seconds (Target: < 2.0s)")

    print("\n" + "=" * 70)
    print(f"✨ EXPORT COMPLETE in {time.time() - t0:.2f}s! READY FOR RENDER DEPLOYMENT.")
    print("=" * 70)


if __name__ == '__main__':
    main()
