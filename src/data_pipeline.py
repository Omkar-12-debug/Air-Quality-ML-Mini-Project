"""
Data Processing & Telemetry Sanitation Pipeline
Urban Atmospheric Telemetry & Machine Learning Command Center
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: List[str] = [
    'CO_True',
    'Sensor_CO',
    'Sensor_NMHC',
    'NOx_True',
    'Sensor_NOx',
    'NO2_True',
    'Sensor_NO2',
    'Sensor_O3',
    'Temperature',
    'Relative_Humidity',
    'Absolute_Humidity'
]

COLUMN_MAPPING: Dict[str, str] = {
    'CO(GT)': 'CO_True',
    'PT08.S1(CO)': 'Sensor_CO',
    'C6H6(GT)': 'Benzene_Target',
    'PT08.S2(NMHC)': 'Sensor_NMHC',
    'NOx(GT)': 'NOx_True',
    'PT08.S3(NOx)': 'Sensor_NOx',
    'NO2(GT)': 'NO2_True',
    'PT08.S4(NO2)': 'Sensor_NO2',
    'PT08.S5(O3)': 'Sensor_O3',
    'T': 'Temperature',
    'RH': 'Relative_Humidity',
    'AH': 'Absolute_Humidity'
}

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    'CO_True': {
        'label': 'CO Concentration (True Reference)',
        'unit': 'mg/m³',
        'desc': 'True Carbon Monoxide concentration measured by reference analyzer.',
        'min': 0.1, 'max': 12.0, 'step': 0.1, 'default': 2.13
    },
    'Sensor_CO': {
        'label': 'Sensor PT08.S1 (CO)',
        'unit': 'mV/Ohm Response',
        'desc': 'Tin oxide semiconductor sensor targeted for CO monitoring.',
        'min': 640.0, 'max': 2100.0, 'step': 10.0, 'default': 1100.0
    },
    'Sensor_NMHC': {
        'label': 'Sensor PT08.S2 (NMHC)',
        'unit': 'mV/Ohm Response',
        'desc': 'Titania chemical sensor calibrated for Non-Metanic Hydrocarbons & Benzene correlation.',
        'min': 380.0, 'max': 2250.0, 'step': 10.0, 'default': 940.0
    },
    'NOx_True': {
        'label': 'NOx Concentration (True Reference)',
        'unit': 'ppb',
        'desc': 'Nitrogen Oxides concentration measured by chemiluminescent reference analyzer.',
        'min': 2.0, 'max': 1500.0, 'step': 5.0, 'default': 240.0
    },
    'Sensor_NOx': {
        'label': 'Sensor PT08.S3 (NOx)',
        'unit': 'mV/Ohm Response',
        'desc': 'Tungsten oxide sensor targeted for NOx sensing.',
        'min': 320.0, 'max': 2700.0, 'step': 10.0, 'default': 830.0
    },
    'NO2_True': {
        'label': 'NO2 Concentration (True Reference)',
        'unit': 'µg/m³',
        'desc': 'Nitrogen Dioxide concentration by reference analyzer.',
        'min': 2.0, 'max': 350.0, 'step': 1.0, 'default': 110.0
    },
    'Sensor_NO2': {
        'label': 'Sensor PT08.S4 (NO2)',
        'unit': 'mV/Ohm Response',
        'desc': 'Tungsten oxide semiconductor sensor tuned for NO2.',
        'min': 550.0, 'max': 2800.0, 'step': 10.0, 'default': 1450.0
    },
    'Sensor_O3': {
        'label': 'Sensor PT08.S5 (O3)',
        'unit': 'mV/Ohm Response',
        'desc': 'Indium oxide sensor tuned for Ozone and photochemical oxidants.',
        'min': 220.0, 'max': 2550.0, 'step': 10.0, 'default': 1030.0
    },
    'Temperature': {
        'label': 'Ambient Air Temperature',
        'unit': '°C',
        'desc': 'Ambient atmospheric temperature at telemetry station.',
        'min': -5.0, 'max': 45.0, 'step': 0.5, 'default': 18.2
    },
    'Relative_Humidity': {
        'label': 'Relative Humidity',
        'unit': '%',
        'desc': 'Atmospheric moisture percentage.',
        'min': 5.0, 'max': 90.0, 'step': 0.5, 'default': 49.2
    },
    'Absolute_Humidity': {
        'label': 'Absolute Humidity',
        'unit': 'g/m³',
        'desc': 'Absolute volumetric moisture content in air.',
        'min': 0.1, 'max': 2.5, 'step': 0.05, 'default': 1.02
    }
}

INFERENCE_PRESETS: Dict[str, Dict[str, float]] = {
    'Clean Morning Air': {
        'CO_True': 0.8,
        'Sensor_CO': 850.0,
        'Sensor_NMHC': 550.0,
        'NOx_True': 55.0,
        'Sensor_NOx': 1350.0,
        'NO2_True': 42.0,
        'Sensor_NO2': 1100.0,
        'Sensor_O3': 600.0,
        'Temperature': 14.5,
        'Relative_Humidity': 62.0,
        'Absolute_Humidity': 0.95
    },
    'Moderate Traffic': {
        'CO_True': 2.1,
        'Sensor_CO': 1080.0,
        'Sensor_NMHC': 920.0,
        'NOx_True': 230.0,
        'Sensor_NOx': 820.0,
        'NO2_True': 110.0,
        'Sensor_NO2': 1460.0,
        'Sensor_O3': 1020.0,
        'Temperature': 18.5,
        'Relative_Humidity': 48.0,
        'Absolute_Humidity': 1.01
    },
    'Severe Smog Alert': {
        'CO_True': 6.2,
        'Sensor_CO': 1680.0,
        'Sensor_NMHC': 1650.0,
        'NOx_True': 780.0,
        'Sensor_NOx': 450.0,
        'NO2_True': 220.0,
        'Sensor_NO2': 2150.0,
        'Sensor_O3': 1850.0,
        'Temperature': 28.0,
        'Relative_Humidity': 32.0,
        'Absolute_Humidity': 1.35
    }
}


def load_and_clean_data(csv_path: str = 'AirQualityUCI.csv') -> pd.DataFrame:
    """
    Loads raw Air Quality UCI semicolon-delimited CSV, handles -200 sentinel missing values,
    prunes metadata and heavily sparse channels, applies linear interpolation with boundary fill,
    standardizes column taxonomy, and computes dual regression/classification targets.
    """
    if not os.path.exists(csv_path):
        # Look in current or parent directory
        alt_paths = [
            os.path.join(os.path.dirname(__file__), '..', csv_path),
            os.path.join('.', csv_path),
            os.path.join('..', csv_path)
        ]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                csv_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset not located at '{csv_path}'.")

    # Read CSV with semicolon delimiter and comma decimal separator
    df = pd.read_csv(csv_path, sep=';', decimal=',', na_values=-200)

    # Strip completely blank rows and trailing separator columns
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

    # Drop non-predictive timestamps and heavily missing NMHC(GT) column
    df_clean = df.drop(columns=['Date', 'Time', 'NMHC(GT)'], errors='ignore').copy()

    # Continuous time-series linear interpolation followed by boundary fill
    df_clean = df_clean.interpolate(method='linear').bfill().ffill()

    # Standardize column nomenclature
    df_clean = df_clean.rename(columns=COLUMN_MAPPING)

    # Formulate binary classification target: 1 if Benzene > 10.0 ug/m3 else 0
    df_clean['Air_Quality_Category'] = (df_clean['Benzene_Target'] > 10.0).astype(int)

    return df_clean


def split_and_scale_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Partitions the cleaned dataset into 80/20 train/test splits stratified by the binary target
    and applies standard scaling across the 11 telemetry features.
    """
    X = df[FEATURE_COLS].copy()
    y_reg = df['Benzene_Target'].copy()
    y_clf = df['Air_Quality_Category'].copy()

    # Stratified split ensures equal class representation in both sets
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLS,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLS,
        index=X_test.index
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_clf_train': y_clf_train,
        'y_clf_test': y_clf_test,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS
    }
