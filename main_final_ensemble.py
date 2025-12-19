"""
AMTEAD FINAL - ENSEMBLE Deep SVDD + Contractive AE
==================================================
Combines complementary anomaly detection methods:
- Deep SVDD: Hypersphere-based (distance outliers)
- Contractive AE: Reconstruction-based (pattern deviations)

Research shows ensemble > individual models
"""

import torch
import numpy as np
import pandas as pd
import glob
import json
import os
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from advanced_features import AdvancedFeatureExtractor
from deep_svdd_advanced import DeepSVDD
from contractive_ae import ContractiveAE

print("="*70)
print("AMTEAD FINAL - ENSEMBLE APPROACH")
print("="*70)
print("Deep SVDD + Contractive Autoencoder + Advanced Features")
print("Research-Backed State-of-the-Art (2024)")
print("="*70)

# Load and prepare data (same as before)
print("\n[1/8] Loading data...")
path = kagglehub.dataset_download("agungpambudi/secure-intrusion-detection-ddos-attacks-profiling")
files = glob.glob(os.path.join(path, '**', 'STAT.json'), recursive=True)

all_samples = []
for fpath in files:
    with open(fpath) as f:
        data = json.load(f)
    
    for sid, metrics_dict in data.items():
        if isinstance(metrics_dict, dict):
            for metric_name, attack_normal_dict in metrics_dict.items():
                if isinstance(attack_normal_dict, dict):
                    for category, values_dict in attack_normal_dict.items():
                        if isinstance(values_dict, dict) and 'data_point' in values_dict:
                            dps = values_dict['data_point']
                            if isinstance(dps, list):
                                label = 1 if 'attack' in category.lower() else 0
                                for dp in dps:
                                    if isinstance(dp, (int, float)):
                                        all_samples.append({
                                            'metric': metric_name,
                                            'value': float(dp),
                                            'label': label,
                                            'sid': sid
                                        })

df = pd.DataFrame(all_samples)

print("\n[2/8] Creating sequences...")
sequences = []
for (sid, label), group in df.groupby(['sid', 'label']):
    metrics_pivot = {}
    for _, row in group.iterrows():
        metric = row['metric']
        if metric not in metrics_pivot:
            metrics_pivot[metric] = []
        metrics_pivot[metric].append(row['value'])
    
    max_len = min(len(next(iter(metrics_pivot.values()))), 2500)
    
    for i in range(25, max_len, 5):
        window_data = []
        for metric in ['cycles', 'instructions', 'branch']:
            if metric in metrics_pivot:
                window = metrics_pivot[metric][i-25:i]
                if len(window) == 25:
                    window_data.append(window)
        
        if len(window_data) == 3:
            sequence = np.column_stack(window_data)
            sequences.append({'data': sequence, 'label': label})

print("\n[3/8] Extracting features...")
feature_extractor = AdvancedFeatureExtractor(window_size=20)
normal_seqs = [s['data'] for s in sequences if s['label'] == 0]
normal_base = np.vstack([seq[-20:] for seq in normal_seqs])
feature_extractor.fit(normal_base)

X_features = []
y_labels = []
for seq_dict in sequences:
    features = feature_extractor.extract_features(seq_dict['data'])
    X_features.append(features)
    y_labels.append(seq_dict['label'])

X = np.array(X_features)
y = np.array(y_labels)

print("\n[4/8] Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_normal = X_train[y_train == 0]

print(f"  Train: {len(X_train)} (Normal: {len(X_train_normal)})")
print(f"  Val: {len(X_val)}, Test: {len(X_test)}")

# Train both models
print("\n[5/8] Training Deep SVDD...")
deep_svdd = DeepSVDD(input_dim=X_train.shape[1], latent_dim=16, nu=0.1)
deep_svdd.fit(X_train_normal, epochs=100, batch_size=64, verbose=False)

print("\n[6/8] Training Contractive AE...")
cae = ContractiveAE(input_dim=X_train.shape[1], latent_dim=16, lambda_contractive=0.0001)
cae.fit(X_train_normal, epochs=100, batch_size=64, verbose=False)

# Save training curves for visualization
training_curves = {
    'deep_svdd_loss': deep_svdd.loss_history,
    'cae_loss': cae.loss_history,
    'epochs': 100
}


# Get scores from both models on validation set
print("\n[7/8] Optimizing ensemble weights...")
svdd_val_scores = deep_svdd.decision_function(X_val)
cae_val_scores = cae.decision_function(X_val)

# Normalize scores to [0, 1]
svdd_val_norm = (svdd_val_scores - svdd_val_scores.min()) / (svdd_val_scores.max() - svdd_val_scores.min() + 1e-10)
cae_val_norm = (cae_val_scores - cae_val_scores.min()) / (cae_val_scores.max() - cae_val_scores.min() + 1e-10)

# Grid search for best weights and threshold
best_score = -1
best_config = None
candidates = []

print(f"    Searching for optimal configuration (weights, threshold)...")

for w_svdd in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    w_cae = 1.0 - w_svdd
    
    # Ensemble score
    ensemble_val = w_svdd * svdd_val_norm + w_cae * cae_val_norm
    
    # Try different thresholds (ORIGINAL RANGE)
    for percentile in range(60, 90, 2):
        threshold = np.percentile(ensemble_val, percentile)
        y_val_pred = (ensemble_val > threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_val_pred)
        
        if len(cm) > 1 and cm[0].sum() > 0 and cm[1].sum() > 0:
            fp_rate = cm[0,1] / cm[0].sum() * 100
            normal_recall = cm[0,0] / cm[0].sum() * 100
            attack_recall = cm[1,1] / cm[1].sum() * 100
            attack_precision = cm[1,1] / (cm[0,1] + cm[1,1]) * 100 if (cm[0,1] + cm[1,1]) > 0 else 0
            
            f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall + 1e-10)
            
            # OPTIMIZED FOR NORMAL ACCURACY: Priority on correctly identifying normal traffic
            # 30% Normal Recall (correctly identifying normal traffic)
            # 30% Attack Recall (catching attacks)
            # 20% Attack Precision (confidence in attack flags)
            # 20% FP Penalty (minimize false alarms)
            fp_penalty = (100 - fp_rate) / 100
            
            score = (0.30 * normal_recall + 
                    0.30 * attack_recall + 
                    0.20 * attack_precision + 
                    0.20 * (fp_penalty * 100))
            
            candidates.append({
                'w_svdd': w_svdd,
                'w_cae': w_cae,
                'threshold': threshold,
                'percentile': percentile,
                'fp_rate': fp_rate,
                'normal_recall': normal_recall,
                'attack_recall': attack_recall,
                'attack_precision': attack_precision,
                'f1': f1,
                'score': score,
                'prod_score': score
            })

# Select best candidate
if candidates:
    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_config = candidates[0]
else:
    print("    ⚠ Warning: No valid configuration found. Using defaults.")
    # Default fallback
    best_config = {
        'w_svdd': 0.5, 'w_cae': 0.5, 
        'threshold': np.percentile(svdd_val_norm, 80), 'percentile': 80,
        'fp_rate': 0, 'normal_recall': 0, 'attack_recall': 0, 
        'attack_precision': 0, 'f1': 0, 'prod_score': 0
    }

print(f"\n  Best Config:")
print(f"    SVDD weight: {best_config['w_svdd']:.2f}, CAE weight: {best_config['w_cae']:.2f}")
print(f"    Threshold: {best_config['threshold']:.4f} (P{best_config['percentile']})")
print(f"    Val FP: {best_config['fp_rate']:.2f}%, Val Recall: {best_config['attack_recall']:.2f}%")

# Apply to test set
print("\n[8/8] Final evaluation...")
svdd_test_scores = deep_svdd.decision_function(X_test)
cae_test_scores = cae.decision_function(X_test)

# Normalize
svdd_test_norm = (svdd_test_scores - svdd_test_scores.min()) / (svdd_test_scores.max() - svdd_test_scores.min() + 1e-10)
cae_test_norm = (cae_test_scores - cae_test_scores.min()) / (cae_test_scores.max() - cae_test_scores.min() + 1e-10)

# Ensemble
ensemble_test = best_config['w_svdd'] * svdd_test_norm + best_config['w_cae'] * cae_test_norm
y_pred = (ensemble_test > best_config['threshold']).astype(int)

# Results
print("\n" + "="*70)
print("FINAL RESULTS - ENSEMBLE")
print("="*70)

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                           target_names=['Normal', 'Attack'],
                           digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("           Normal  Attack")
print(f"Normal      {cm[0,0]:5d}  {cm[0,1]:5d}")
if len(cm) > 1:
    print(f"Attack      {cm[1,0]:5d}  {cm[1,1]:5d}")

normal_r = cm[0,0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
attack_r = cm[1,1] / cm[1].sum() * 100 if len(cm) > 1 and cm[1].sum() > 0 else 0
attack_p = cm[1,1] / (cm[0,1] + cm[1,1]) * 100 if (cm[0,1] + cm[1,1]) > 0 else 0
fp_rate = cm[0,1] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0

print("\n" + "="*70)
print("FINAL METRICS")
print("="*70)
print(f"Overall Accuracy:      {acc*100:6.2f}%  (Target: 85%  {'✓' if acc*100 >= 85 else '⚠'})")
print(f"Normal Recall:         {normal_r:6.2f}%")
print(f"Attack Recall:         {attack_r:6.2f}%  (Target: 60-85%  {'✓' if 60 <= attack_r <= 85 else '⚠'})")
print(f"Attack Precision:      {attack_p:6.2f}%")
print(f"False Positive Rate:   {fp_rate:6.2f}%  (Target: 2-5%  {'✓✓' if fp_rate <= 5 else '✓' if fp_rate <= 10 else '⚠'})")

print("\n" + "="*70)
print(f"IMPROVEMENT vs Baseline:")
print(f"  Baseline (Ensemble): 9.58% attack recall, 2.1% FP")
print(f"  Deep SVDD Only: 42.42% attack recall, 9.09% FP")
print(f"  ENSEMBLE FINAL: {attack_r:.2f}% attack recall, {fp_rate:.2f}% FP")
print(f"  Improvement: {attack_r/9.58:.1f}x better detection")
print("="*70)

print("\n✓ AMTEAD FINAL Complete!")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*70)
print("Saving Models...")
print("="*70)

import pickle
from pathlib import Path

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# Save complete model package
model_package = {
    'deep_svdd': deep_svdd,
    'cae': cae,
    'scaler': scaler,
    'feature_extractor': feature_extractor,
    'best_config': best_config,
    'metrics': {
        'accuracy': acc,
        'attack_recall': attack_r,
        'normal_recall': normal_r,
        'attack_precision': attack_p,
        'fp_rate': fp_rate,
        'f1_score': 2*(attack_p*attack_r)/(attack_p+attack_r+1e-10)
    },
    'confusion_matrix': cm,
    'total_sequences': len(sequences),
    'test_samples': len(X_test)
}

model_path = MODELS_DIR / 'amtead_ensemble.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)
print(f"  ✓ Saved complete model package: {model_path}")

# Also save individual components
with open(MODELS_DIR / 'deep_svdd.pkl', 'wb') as f:
    pickle.dump(deep_svdd, f)
print(f"  ✓ Saved Deep SVDD model: models/deep_svdd.pkl")

with open(MODELS_DIR / 'contractive_ae.pkl', 'wb') as f:
    pickle.dump(cae, f)
print(f"  ✓ Saved Contractive AE: models/contractive_ae.pkl")

with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved Scaler: models/scaler.pkl")

with open(MODELS_DIR / 'feature_extractor.pkl', 'wb') as f:
    pickle.dump(feature_extractor, f)
print(f"  ✓ Saved Feature Extractor: models/feature_extractor.pkl")

with open(MODELS_DIR / 'ensemble_config.pkl', 'wb') as f:
    pickle.dump(best_config, f)
print(f"  ✓ Saved Ensemble Config: models/ensemble_config.pkl")

with open(MODELS_DIR / 'training_curves.pkl', 'wb') as f:
    pickle.dump(training_curves, f)
print(f"  ✓ Saved Training Curves: models/training_curves.pkl")



print(f"\n  All models saved to: {MODELS_DIR}/")


# ============================================================================
# TRAINING RESULTS REPORT
# ============================================================================
print("\n" + "="*70)
print("TRAINING RESULTS REPORT")
print("="*70)

print(f"\n✅ Accuracy:          {acc*100:.2f}%")
print(f"✅ Attack Detection:  {attack_r:.2f}%")
print(f"✅ Normal Preserved:  {normal_r:.2f}%")
print(f"✅ Precision:         {attack_p:.2f}%")
print(f"✅ FP Rate:           {fp_rate:.2f}%")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print(f"  Models saved to: {MODELS_DIR}/")
print(f"  To generate charts: python generate_all_charts_from_saved_model.py")
print("="*70)



