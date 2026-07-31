"""
Analysis script to examine Fold 3 drop in performance,
generate confusion matrices, and save ROC and Precision-Recall curves.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GroupKFold

# Ensure directories exist
os.makedirs('plots', exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
df_ml = pd.read_csv('data/ml_dataset.csv')
df_orig = pd.read_csv('data/training_data.csv')

# Sort original data exactly like ML dataset creation did
df_orig = df_orig.sort_values(['Race', 'Driver', 'LapNumber']).reset_index(drop=True)

if len(df_ml) != len(df_orig):
    print(f"Warning: df_ml length ({len(df_ml)}) does not match df_orig length ({len(df_orig)}).")
    # Align by index
    df_orig = df_orig.iloc[:len(df_ml)]

# Features and target
feature_cols = [c for c in df_ml.columns if c not in ('Race', 'PitNextLap')]
X = df_ml[feature_cols]
y = df_ml['PitNextLap'].values
race_groups = df_ml['Race']

race_to_group = {race: idx for idx, race in enumerate(df_ml['Race'].unique())}
groups = race_groups.map(race_to_group).values

gkf = GroupKFold(n_splits=5)

# ============================================================================
# RUN CV AND ACCUMULATE METRICS
# ============================================================================

fold_results = {}
all_y_test = []
all_y_prob = []
all_y_pred = []

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Store predictions and targets for overall metrics
    all_y_test.extend(y_test)
    all_y_prob.extend(y_prob)
    all_y_pred.extend(y_pred)
    
    # Metrics
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    fold_results[fold] = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "races": race_groups.iloc[test_idx].unique().tolist()
    }
    
    # Plot ROC curve for fold
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.3f})')
    
    # Plot PR curve for fold
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax_pr.plot(recall_vals, precision_vals, label=f'Fold {fold} (AP = {ap:.3f})')

# Overall confusion matrix
overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(all_y_test, all_y_pred).ravel()

# Overall curves
overall_fpr, overall_tpr, _ = roc_curve(all_y_test, all_y_prob)
overall_auc = auc(overall_fpr, overall_tpr)
ax_roc.plot(overall_fpr, overall_tpr, 'k--', label=f'Overall Baseline (AUC = {overall_auc:.3f})', linewidth=2)

overall_prec_vals, overall_rec_vals, _ = precision_recall_curve(all_y_test, all_y_prob)
overall_ap = average_precision_score(all_y_test, all_y_prob)
ax_pr.plot(overall_rec_vals, overall_prec_vals, 'k--', label=f'Overall Baseline (AP = {overall_ap:.3f})', linewidth=2)

# Save ROC plot
ax_roc.set_title('ROC Curves - Random Forest Baseline')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(loc='lower right')
ax_roc.grid(True, linestyle='--', alpha=0.5)
fig_roc.tight_layout()
fig_roc.savefig('plots/roc_curves.png', dpi=300)
plt.close(fig_roc)

# Save PR plot
ax_pr.set_title('Precision-Recall Curves - Random Forest Baseline')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.legend(loc='lower left')
ax_pr.grid(True, linestyle='--', alpha=0.5)
fig_pr.tight_layout()
fig_pr.savefig('plots/precision_recall_curves.png', dpi=300)
plt.close(fig_pr)

# ============================================================================
# FOLD 3 DETAILED ANALYSIS
# ============================================================================

print("\nRunning Fold 3 detailed analysis...")
splits = list(gkf.split(X, y, groups))
fold3_train_idx, fold3_test_idx = splits[2] # 0-indexed split 2 is fold 3

X_train_3, X_test_3 = X.iloc[fold3_train_idx], X.iloc[fold3_test_idx]
y_train_3, y_test_3 = y[fold3_train_idx], y[fold3_test_idx]

rf3 = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf3.fit(X_train_3, y_train_3)
fold3_preds = rf3.predict(X_test_3)
fold3_probs = rf3.predict_proba(X_test_3)[:, 1]

# Align predictions with original dataframe to study raw characteristics
df_fold3_orig = df_orig.iloc[fold3_test_idx].copy()
df_fold3_orig['Pred_PitNextLap'] = fold3_preds
df_fold3_orig['Pred_Prob'] = fold3_probs

fold3_race_stats = []
for race in df_fold3_orig['Race'].unique():
    race_df = df_fold3_orig[df_fold3_orig['Race'] == race]
    r_y = race_df['PitNextLap'].values
    r_pred = race_df['Pred_PitNextLap'].values
    
    r_tn, r_fp, r_fn, r_tp = confusion_matrix(r_y, r_pred, labels=[0, 1]).ravel()
    r_prec = precision_score(r_y, r_pred, zero_division=0)
    r_rec = recall_score(r_y, r_pred, zero_division=0)
    r_f1 = f1_score(r_y, r_pred, zero_division=0)
    
    # Identify track issues/yellow flags/safety cars (TrackStatus standard definitions:
    # 1: Green, 2: Yellow, 4: Safety Car, 8: Red Flag, etc. Often combined as digits in strings, e.g. "4", "26", "671")
    # Let's count laps containing safety car (4), virtual safety car (6 or sometimes 5/7 depending on raw codes,
    # let's look for anything other than '1' which is green)
    non_green_laps = race_df[race_df['TrackStatus'].astype(str) != '1'].shape[0]
    sc_vsc_laps = race_df[race_df['TrackStatus'].astype(str).str.contains('[4567]')].shape[0]
    total_laps = len(race_df)
    
    # Count wet compound laps (Intermediates or Wet)
    wet_compounds = race_df[race_df['Compound'].astype(str).str.upper().isin(['INTERMEDIATE', 'WET', 'INTERS'])]
    wet_laps = len(wet_compounds)
    
    # Print race summary
    print(f"\nRace: {race}")
    print(f"  Laps: {total_laps} | Pits: {r_y.sum()} | TP: {r_tp} | FP: {r_fp} | FN: {r_fn} | TN: {r_tn}")
    print(f"  Precision: {r_prec:.3f} | Recall: {r_rec:.3f} | F1: {r_f1:.3f}")
    print(f"  Non-green status laps: {non_green_laps} | SC/VSC laps: {sc_vsc_laps} | Wet laps: {wet_laps}")
    
    # List actual cases of FN and FP
    # Let's see some samples
    fn_df = race_df[(race_df['PitNextLap'] == 1) & (race_df['Pred_PitNextLap'] == 0)]
    fp_df = race_df[(race_df['PitNextLap'] == 0) & (race_df['Pred_PitNextLap'] == 1)]
    
    fold3_race_stats.append({
        "Race": race,
        "TotalLaps": int(total_laps),
        "ActualPits": int(r_y.sum()),
        "TP": int(r_tp),
        "FP": int(r_fp),
        "FN": int(r_fn),
        "TN": int(r_tn),
        "Precision": float(r_prec),
        "Recall": float(r_rec),
        "F1": float(r_f1),
        "NonGreenLaps": int(non_green_laps),
        "SCVSCLaps": int(sc_vsc_laps),
        "WetLaps": int(wet_laps),
        "FNSamples": fn_df[['Driver', 'LapNumber', 'Stint', 'TyreLife', 'Compound', 'TrackStatus', 'Pred_Prob']].head(3).to_dict('records'),
        "FPSamples": fp_df[['Driver', 'LapNumber', 'Stint', 'TyreLife', 'Compound', 'TrackStatus', 'Pred_Prob']].head(3).to_dict('records')
    })

# ============================================================================
# SAVE RESULTS JSON
# ============================================================================

final_results = {
    "fold_metrics": fold_results,
    "overall_cm": {
        "TN": int(overall_tn),
        "FP": int(overall_fp),
        "FN": int(overall_fn),
        "TP": int(overall_tp)
    },
    "fold3_race_stats": fold3_race_stats
}

with open('baseline_analysis_metrics.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n" + "="*70)
print("ANALYSIS PROCESS COMPLETE")
print("="*70)
print(f"Overall confusion matrix: TN={overall_tn}, FP={overall_fp}, FN={overall_fn}, TP={overall_tp}")
print("Saved baseline_analysis_metrics.json")
print("Saved plots/roc_curves.png")
print("Saved plots/precision_recall_curves.png")
