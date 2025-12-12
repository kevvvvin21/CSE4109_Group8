"""
XGBoost Prediction Script (Simple Version)
Make predictions using XGBoost model saved with joblib

Usage:
    python predict_xgboost_simple.py

Requirements:
    pip install xgboost scikit-learn pandas numpy joblib
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from datetime import datetime

print("="*70)
print("XGBOOST - PREDICTION SCRIPT (Simple Version)")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Path to your trained XGBoost model
MODEL_PATH = 'xgboost_model.pkl'

# Path to your new data file
DATA_PATH = 'data_with_split.csv'

# Optional: Filter data by split value (0=train, 1=val, 2=test, None=all)
FILTER_SPLIT = 2  # Predict on test set only

# Prediction threshold (default: 0.5)
THRESHOLD = 0.5

# =============================================================================
# LOAD MODEL
# =============================================================================
print("Step 1: Loading trained model...")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"  • Model type: {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"  • Number of trees: {model.n_estimators}\n")
    
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    print("   Please train the model first using train_xgboost_simple.py")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Step 2: Loading data...")

try:
    df = pd.read_csv(DATA_PATH, low_memory=False, on_bad_lines='skip')
    
    # Clean data
    df = df[pd.to_numeric(df['split'], errors='coerce').notna()]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # Filter by split if specified
    if FILTER_SPLIT is not None:
        df = df[df['split'] == FILTER_SPLIT].copy()
        print(f"✓ Data loaded and filtered (split={FILTER_SPLIT}): {len(df)} samples")
    else:
        print(f"✓ Data loaded: {len(df)} samples")
    
    # Check if labels exist
    has_labels = 'label' in df.columns
    if has_labels:
        print(f"✓ Ground truth labels found - will calculate accuracy")
    else:
        print(f"ℹ No labels found - will only generate predictions")
    
    print()
    
except FileNotFoundError:
    print(f"❌ Error: Data file not found at {DATA_PATH}")
    exit(1)
except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    exit(1)

# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("Step 3: Preparing features...")

# Get feature columns (exclude label and split)
feature_columns = [col for col in df.columns if col not in ['label', 'split']]

X = df[feature_columns]
print(f"✓ Features prepared: {len(feature_columns)} features")
print(f"  Feature names: {feature_columns[:3]}... (showing first 3)\n")

# =============================================================================
# MAKE PREDICTIONS
# =============================================================================
print("="*70)
print("Step 4: Making predictions")
print("="*70)

# Predict probabilities
probabilities = model.predict_proba(X)[:, 1]

# Convert to class predictions using threshold
predictions = (probabilities > THRESHOLD).astype(int)

print(f"✓ Predictions complete for {len(predictions)} samples")
print(f"  Prediction threshold: {THRESHOLD}")
print(f"\nPrediction Summary:")
print(f"  • Class 0: {sum(predictions == 0)} samples ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
print(f"  • Class 1: {sum(predictions == 1)} samples ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
print(f"  • Average probability: {probabilities.mean():.4f}")
print(f"  • Min probability: {probabilities.min():.4f}")
print(f"  • Max probability: {probabilities.max():.4f}")
print()

# =============================================================================
# EVALUATE (if labels available)
# =============================================================================
if has_labels:
    print("="*70)
    print("Step 5: Evaluation Results")
    print("="*70)
    
    y_true = df['label'].astype(int).values
    
    accuracy = accuracy_score(y_true, predictions)
    auroc = roc_auc_score(y_true, probabilities)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUROC:    {auroc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_true, predictions, target_names=['Class 0', 'Class 1']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, predictions)
    print(f"                Predicted")
    print(f"              Class 0  Class 1")
    print(f"Actual  0      {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"        1      {cm[1][0]:6d}  {cm[1][1]:6d}")
    print()

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================
print("="*70)
print("Step 6: Saving predictions")
print("="*70)

# Create results dataframe
results = pd.DataFrame({
    'predicted_class': predictions,
    'predicted_probability': probabilities
})

# Add actual labels if available
if has_labels:
    results.insert(0, 'actual_label', y_true)
    results['correct'] = (results['actual_label'] == results['predicted_class'])

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'xgb_predictions_{timestamp}.csv'
results.to_csv(output_file, index=False)

print(f"✓ Predictions saved to: {output_file}")
print(f"  Total samples: {len(results)}")

# =============================================================================
# SHOW SAMPLE PREDICTIONS
# =============================================================================
print("\n" + "="*70)
print("Sample Predictions (first 20 rows)")
print("="*70)
print(results.head(20).to_string(index=False))

# =============================================================================
# STATISTICS
# =============================================================================
if has_labels:
    print("\n" + "="*70)
    print("Prediction Statistics")
    print("="*70)
    
    # Correct predictions by class
    class_0_correct = sum((y_true == 0) & (predictions == 0))
    class_0_total = sum(y_true == 0)
    class_1_correct = sum((y_true == 1) & (predictions == 1))
    class_1_total = sum(y_true == 1)
    
    print(f"\nClass 0 Performance:")
    print(f"  • Correct: {class_0_correct}/{class_0_total} ({class_0_correct/class_0_total*100:.1f}%)")
    
    print(f"\nClass 1 Performance:")
    print(f"  • Correct: {class_1_correct}/{class_1_total} ({class_1_correct/class_1_total*100:.1f}%)")
    
    # Confidence analysis
    high_confidence = sum((probabilities > 0.8) | (probabilities < 0.2))
    medium_confidence = sum((probabilities >= 0.4) & (probabilities <= 0.6))
    
    print(f"\nConfidence Analysis:")
    print(f"  • High confidence (>0.8 or <0.2): {high_confidence} samples ({high_confidence/len(probabilities)*100:.1f}%)")
    print(f"  • Uncertain (0.4-0.6): {medium_confidence} samples ({medium_confidence/len(probabilities)*100:.1f}%)")
    
    # Threshold analysis
    print(f"\nThreshold Analysis (Current: {THRESHOLD}):")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_at_thresh = (probabilities > thresh).astype(int)
        acc_at_thresh = accuracy_score(y_true, pred_at_thresh)
        recall_1 = sum((y_true == 1) & (pred_at_thresh == 1)) / max(sum(y_true == 1), 1)
        precision_1 = sum((y_true == 1) & (pred_at_thresh == 1)) / max(sum(pred_at_thresh == 1), 1)
        print(f"  • Threshold {thresh:.1f}: Acc={acc_at_thresh:.3f}, Recall(1)={recall_1:.3f}, Prec(1)={precision_1:.3f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PREDICTION COMPLETE!")
print("="*70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\nConfiguration:")
print(f"  • Model: {MODEL_PATH}")
print(f"  • Data: {DATA_PATH}")
print(f"  • Threshold: {THRESHOLD}")

print(f"\nGenerated files:")
print(f"  • {output_file}")

if has_labels:
    print(f"\nFinal Performance:")
    print(f"  • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • AUROC: {auroc:.4f}")

print("\n" + "="*70)
print("\nTip: Adjust THRESHOLD in the script to change sensitivity")
print("  • Lower threshold (0.3) → Catch more Class 1 (higher recall)")
print("  • Higher threshold (0.7) → Fewer false alarms (higher precision)")
print("="*70)