"""
Random Forest baseline with GroupKFold CV, adapted for f1-strategist-master.

Input:  data/ml_dataset.csv (features + Race + PitNextLap, no separate labels.csv)
Output: baseline_rf_v2.pkl, baseline_results_v2.json
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data/ml_dataset.csv...")
df = pd.read_csv('data/ml_dataset.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# PitNextLap is the label, Race is the grouping column — everything else is a feature
feature_cols = [c for c in df.columns if c not in ('Race', 'PitNextLap')]
X = df[feature_cols]
y = df['PitNextLap'].values
race_groups = df['Race']

print(f"Features ({len(feature_cols)}): {feature_cols}\n")
print(f"Class distribution:")
print(f"  Non-pit (0): {(y == 0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")
print(f"  Pit (1):     {(y == 1).sum()} ({100*(y==1).sum()/len(y):.1f}%)\n")

# ============================================================================
# GROUPKFOLD CV
# ============================================================================

race_to_group = {race: idx for idx, race in enumerate(df['Race'].unique())}
groups = race_groups.map(race_to_group).values

gkf = GroupKFold(n_splits=5)
n_races = df['Race'].nunique()
print(f"Total races: {n_races} | ~{n_races/5:.1f} races per fold\n")

fold_metrics = []
best_model = None
best_f1 = 0
best_fold = None

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"{'='*70}\nFOLD {fold}\n{'='*70}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)} laps | Test: {len(X_test)} laps")
    print(f"Train pit rate: {y_train.mean()*100:.1f}% | Test pit rate: {y_test.mean()*100:.1f}%\n")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("Training...", end=" ")
    rf.fit(X_train, y_train)
    print("[OK]")

    y_pred = rf.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fold_metrics.append({'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1})
    print(f"\nPrecision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}\n")

    if f1 > best_f1:
        best_f1 = f1
        best_model = rf
        best_fold = fold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")

# ============================================================================
# SUMMARY
# ============================================================================

metrics_df = pd.DataFrame(fold_metrics)

print("="*70)
print("CROSS-VALIDATION SUMMARY (5 Folds)")
print("="*70)
print(metrics_df.to_string(index=False))
print(f"\nMean F1: {metrics_df['f1'].mean():.3f} ± {metrics_df['f1'].std():.3f}")
print(f"Best model: Fold {best_fold} (F1 = {best_f1:.3f})")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)
print(importance_df.to_string(index=False))

top_5 = importance_df.head(5)['Feature'].tolist()
if 'LapTimeDelta' in top_5:
    print(f"\n[OK] LapTimeDelta ranks #{top_5.index('LapTimeDelta')+1} in feature importance")

# ============================================================================
# SAVE
# ============================================================================

joblib.dump(best_model, 'baseline_rf_v2.pkl')
print("\n[OK] Model saved: baseline_rf_v2.pkl")

results = {
    'cross_val_metrics': metrics_df.to_dict('records'),
    'mean_precision': float(metrics_df['precision'].mean()),
    'mean_recall': float(metrics_df['recall'].mean()),
    'mean_f1': float(metrics_df['f1'].mean()),
    'std_f1': float(metrics_df['f1'].std()),
    'feature_importances': dict(zip(feature_cols, best_model.feature_importances_.tolist())),
    'top_5_features': top_5
}

with open('baseline_results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print("[OK] Results saved: baseline_results_v2.json")

print("\n" + "="*70)
print("DONE — Next: permutation importance, SMOTE, XGBoost")
print("="*70)
