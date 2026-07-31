"""
Feature preparation with LapTimeDelta, adapted for f1-strategist-master.

Input:  data/training_data.csv (must contain PitNextLap column)
Output: data/ml_dataset.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data/training_data.csv...")
df = pd.read_csv('data/training_data.csv')

print(f"Input shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

required_cols = ['Driver', 'Race', 'LapNumber', 'LapTime', 'Compound',
                  'TyreLife', 'Position', 'TrackStatus', 'PitNextLap']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.sort_values(['Race', 'Driver', 'LapNumber']).reset_index(drop=True)
print("Sorted by Race, Driver, LapNumber.\n")

# ============================================================================
# CONVERT LAPTIME TO SECONDS (if needed)
# ============================================================================

if df['LapTime'].dtype == 'object':
    print("Converting LapTime to seconds...")
    df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

print(f"LapTime range: {df['LapTime'].min():.2f}s – {df['LapTime'].max():.2f}s\n")

# ============================================================================
# CALCULATE LAPTIMEDELTA
# ============================================================================

print("Calculating LapTimeDelta...")

def calc_delta(group):
    group = group.copy()
    group['LapTimeDelta'] = group['LapTime'].diff()
    return group

df = df.groupby(['Race', 'Driver'], group_keys=False).apply(calc_delta)

print(df['LapTimeDelta'].describe())
print(f"Missing (first lap of stint): {df['LapTimeDelta'].isna().sum()}\n")

df['LapTimeDelta'] = df.groupby(['Race', 'Driver'])['LapTimeDelta'].fillna(method='bfill')
df['LapTimeDelta'] = df['LapTimeDelta'].fillna(0.0)

print(f"LapTimeDelta range: {df['LapTimeDelta'].min():.3f}s – {df['LapTimeDelta'].max():.3f}s\n")

# ============================================================================
# ENCODE CATEGORICALS
# ============================================================================

print("Encoding Compound and TrackStatus...")

compound_encoder = LabelEncoder()
df['Compound'] = compound_encoder.fit_transform(df['Compound'].astype(str))
print(f"Compound mapping: {dict(zip(compound_encoder.classes_, compound_encoder.transform(compound_encoder.classes_)))}")

trackstatus_encoder = LabelEncoder()
df['TrackStatus'] = trackstatus_encoder.fit_transform(df['TrackStatus'].astype(str))
print(f"TrackStatus mapping: {dict(zip(trackstatus_encoder.classes_, trackstatus_encoder.transform(trackstatus_encoder.classes_)))}\n")

# ============================================================================
# BUILD FEATURE SET
# ============================================================================

feature_cols = [
    'LapTime', 'LapTimeDelta', 'TyreLife', 'Compound',
    'TrackStatus', 'Position', 'LapNumber', 'Stint',
]

optional_cols = ['Sector1', 'Sector2', 'Sector3', 'AirTemp', 'TrackTemp', 'GapToLeader', 'DRS']
for col in optional_cols:
    if col in df.columns:
        feature_cols.append(col)
        print(f"[OK] Including optional feature: {col}")

print(f"\nUsing {len(feature_cols)} features: {feature_cols}\n")

# Keep Race (for GroupKFold) and PitNextLap (the label) alongside features
ml_features = df[feature_cols + ['Race', 'PitNextLap']].copy()

# Fill any remaining NaN
missing_counts = ml_features[feature_cols].isna().sum()
if missing_counts.any():
    print("Filling missing values...")
    for col in feature_cols:
        if ml_features[col].isna().sum() > 0:
            ml_features[col] = ml_features.groupby(['Race'])[col].fillna(method='ffill')
    ml_features = ml_features.fillna(0)

# ============================================================================
# SAVE
# ============================================================================

import os
os.makedirs('data', exist_ok=True)

output_path = 'data/ml_dataset.csv'
ml_features.to_csv(output_path, index=False)

print(f"[OK] Saved: {output_path}")
print(f"Shape: {ml_features.shape}")
print(f"\nFirst 5 rows:")
print(ml_features.head())

print("\n" + "="*70)
print("FEATURE PREPARATION COMPLETE — LapTimeDelta integrated")
print("="*70)
