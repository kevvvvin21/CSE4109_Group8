"""
XGBoost Model Training Script (Sklearn API)
Train and evaluate an XGBoost classifier using sklearn-compatible API

Usage:
    python train_xgboost_simple.py

Requirements:
    pip install xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("XGBOOST MODEL TRAINING (Simple Version)")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("Step 1: Loading and preparing data...")

# UPDATE THIS PATH TO YOUR CSV FILE
DATA_PATH = 'data_with_split.csv'

df = pd.read_csv(DATA_PATH, low_memory=False, on_bad_lines='skip')

# Clean data
df = df[pd.to_numeric(df['split'], errors='coerce').notna()]
df = df[pd.to_numeric(df['label'], errors='coerce').notna()]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df['label'] = df['label'].astype(int)
df['split'] = df['split'].astype(int)

print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"\nData split:")
print(f"  - Training:   {len(df[df['split']==0])} samples ({len(df[df['split']==0])/len(df)*100:.1f}%)")
print(f"  - Validation: {len(df[df['split']==1])} samples ({len(df[df['split']==1])/len(df)*100:.1f}%)")
print(f"  - Test:       {len(df[df['split']==2])} samples ({len(df[df['split']==2])/len(df)*100:.1f}%)")
print(f"\nClass distribution:")
print(f"  - Class 0: {len(df[df['label']==0])} samples ({len(df[df['label']==0])/len(df)*100:.1f}%)")
print(f"  - Class 1: {len(df[df['label']==1])} samples ({len(df[df['label']==1])/len(df)*100:.1f}%)\n")

# Split data based on 'split' column
train_data = df[df['split'] == 0].copy()
val_data = df[df['split'] == 1].copy()
test_data = df[df['split'] == 2].copy()

# Prepare features and labels
feature_columns = [col for col in df.columns if col not in ['label', 'split']]

X_train = train_data[feature_columns]
y_train = train_data['label']

X_val = val_data[feature_columns]
y_val = val_data['label']

X_test = test_data[feature_columns]
y_test = test_data['label']

print(f"Number of features: {len(feature_columns)}")
print(f"Feature names: {feature_columns[:5]}... (showing first 5)\n")

# =============================================================================
# 2. CONFIGURE AND TRAIN XGBOOST MODEL
# =============================================================================
print("="*70)
print("Step 2: Training XGBoost Model")
print("="*70)

start_time = datetime.now()

# Create XGBoost classifier
model = XGBClassifier(
    n_estimators=1000,           # Maximum number of trees
    max_depth=6,                 # Maximum tree depth
    learning_rate=0.1,           # Step size shrinkage
    subsample=0.8,               # Fraction of samples per tree
    colsample_bytree=0.8,        # Fraction of features per tree
    min_child_weight=1,          # Minimum sum of instance weight
    gamma=0,                     # Minimum loss reduction
    reg_alpha=0,                 # L1 regularization
    reg_lambda=1,                # L2 regularization
    random_state=42,             # Random seed
    eval_metric='auc',           # Evaluation metric
    early_stopping_rounds=50,    # Stop if no improvement
    n_jobs=-1,                   # Use all CPU cores
    verbosity=1                  # Print progress
)

print("\nModel Configuration:")
print("-" * 70)
print(f"  n_estimators:         {model.n_estimators}")
print(f"  max_depth:            {model.max_depth}")
print(f"  learning_rate:        {model.learning_rate}")
print(f"  subsample:            {model.subsample}")
print(f"  colsample_bytree:     {model.colsample_bytree}")
print(f"  early_stopping_rounds: 50")
print()

# Train with validation set for early stopping
print("Training in progress...")
print("-" * 70)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

print("-" * 70)

training_time = (datetime.now() - start_time).total_seconds()

print(f"\n✓ Training complete!")
print(f"  Training time: {training_time:.1f} seconds")
print(f"  Best iteration: {model.best_iteration}")
print(f"  Best validation score: {model.best_score:.4f}")
print(f"  Total trees: {model.n_estimators}\n")

# =============================================================================
# 3. MAKE PREDICTIONS
# =============================================================================
print("="*70)
print("Step 3: Making predictions on all datasets")
print("="*70)

# Predictions on training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]

# Predictions on validation set
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

# Predictions on test set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

print("✓ Predictions complete for all datasets\n")

# =============================================================================
# 4. EVALUATE MODEL
# =============================================================================
print("="*70)
print("Step 4: Model Evaluation")
print("="*70)

def print_evaluation(y_true, y_pred, y_pred_proba, dataset_name):
    """Print detailed evaluation metrics"""
    print(f"\n{dataset_name} SET RESULTS")
    print("-" * 70)
    
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC-ROC:  {auc_roc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
    
    if dataset_name == "TEST":
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Class 0  Class 1")
        print(f"Actual  0      {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"        1      {cm[1][0]:6d}  {cm[1][1]:6d}")
        print()

# Evaluate on all datasets
print_evaluation(y_train, y_train_pred, y_train_pred_proba, "TRAINING")
print_evaluation(y_val, y_val_pred, y_val_pred_proba, "VALIDATION")
print_evaluation(y_test, y_test_pred, y_test_pred_proba, "TEST")

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================
print("="*70)
print("Step 5: Feature Importance Analysis")
print("="*70)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print("-" * 70)
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:8.4f}")
print()

# =============================================================================
# 6. SAVE MODEL
# =============================================================================
print("="*70)
print("Step 6: Saving model and predictions")
print("="*70)

# Save model (using joblib for sklearn-style)
model_path = 'xgboost_model.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved: {model_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'probability': y_test_pred_proba
})
predictions_path = 'xgb_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✓ Predictions saved: {predictions_path}")

# Save feature importance
importance_path = 'xgb_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved: {importance_path}\n")

# =============================================================================
# 7. CREATE VISUALIZATIONS
# =============================================================================
print("="*70)
print("Step 7: Creating visualizations")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Feature Importance (top 15)
top_features = feature_importance.head(15)
axes[0, 0].barh(range(len(top_features)), top_features['importance'], color='steelblue')
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['feature'])
axes[0, 0].set_xlabel('Importance', fontsize=11)
axes[0, 0].set_title('Top 15 Feature Importances', fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
axes[0, 1].set_xlabel('Predicted', fontsize=11)
axes[0, 1].set_ylabel('Actual', fontsize=11)
axes[0, 1].set_title('Test Set Confusion Matrix', fontsize=13, fontweight='bold')

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
auc = roc_auc_score(y_test, y_test_pred_proba)
axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc:.3f})', color='steelblue')
axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 2].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 2].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 2].set_title('ROC Curve (Test Set)', fontsize=13, fontweight='bold')
axes[0, 2].legend(fontsize=10)
axes[0, 2].grid(alpha=0.3)

# 4. Prediction Distribution
axes[1, 0].hist([y_test_pred_proba[y_test == 0], y_test_pred_proba[y_test == 1]],
                bins=30, label=['Class 0', 'Class 1'], alpha=0.7, color=['steelblue', 'coral'])
axes[1, 0].set_xlabel('Predicted Probability', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Test Set Prediction Distribution', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2)

# 5. Performance Comparison Across Datasets
datasets = ['Training', 'Validation', 'Test']
accuracies = [
    accuracy_score(y_train, y_train_pred),
    accuracy_score(y_val, y_val_pred),
    accuracy_score(y_test, y_test_pred)
]
aucs = [
    roc_auc_score(y_train, y_train_pred_proba),
    roc_auc_score(y_val, y_val_pred_proba),
    roc_auc_score(y_test, y_test_pred_proba)
]

x = np.arange(len(datasets))
width = 0.35
axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
axes[1, 1].bar(x + width/2, aucs, width, label='AUC-ROC', alpha=0.8)
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].set_title('Performance Across Datasets', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(datasets)
axes[1, 1].legend(fontsize=10)
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(axis='y', alpha=0.3)

# 6. Class Distribution
class_counts = [sum(y_test == 0), sum(y_test == 1)]
pred_counts = [sum(y_test_pred == 0), sum(y_test_pred == 1)]
x = np.arange(2)
width = 0.35
axes[1, 2].bar(x - width/2, class_counts, width, label='Actual', alpha=0.8)
axes[1, 2].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
axes[1, 2].set_ylabel('Count', fontsize=11)
axes[1, 2].set_title('Class Distribution (Test Set)', fontsize=13, fontweight='bold')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(['Class 0', 'Class 1'])
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.suptitle('XGBoost Model - Comprehensive Analysis',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

viz_path = 'xgboost_results.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Visualization saved: {viz_path}\n")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("="*70)
print("FINAL SUMMARY")
print("="*70)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)
test_precision_0 = precision_score(y_test, y_test_pred, pos_label=0)
test_precision_1 = precision_score(y_test, y_test_pred, pos_label=1)
test_recall_0 = recall_score(y_test, y_test_pred, pos_label=0)
test_recall_1 = recall_score(y_test, y_test_pred, pos_label=1)
test_f1_0 = f1_score(y_test, y_test_pred, pos_label=0)
test_f1_1 = f1_score(y_test, y_test_pred, pos_label=1)

print(f"\nModel Performance on Test Set:")
print(f"  • AUROC:      {test_auc:.4f}")
print(f"  • Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Precision (Class 0): {test_precision_0:.4f}")
print(f"  • Precision (Class 1): {test_precision_1:.4f}")
print(f"  • Recall (Class 0):    {test_recall_0:.4f}")
print(f"  • Recall (Class 1):    {test_recall_1:.4f}")
print(f"  • F1-Score (Class 0):  {test_f1_0:.4f}")
print(f"  • F1-Score (Class 1):  {test_f1_1:.4f}")

print(f"\nModel Information:")
print(f"  • Number of trees: {model.n_estimators}")
print(f"  • Training time: {training_time:.1f} seconds")
print(f"  • Best iteration: {model.best_iteration}")
print(f"  • Number of features: {len(feature_columns)}")

print(f"\nTop 3 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(3).iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

print(f"\nFiles Generated:")
print(f"  1. {model_path} - Trained XGBoost model")
print(f"  2. {predictions_path} - Test set predictions")
print(f"  3. {importance_path} - Feature importance rankings")
print(f"  4. {viz_path} - Comprehensive visualization")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

print("\nTo use this model for predictions:")
print("  import joblib")
print("  model = joblib.load('xgboost_model.pkl')")
print("  predictions = model.predict(X_new)")
print("  probabilities = model.predict_proba(X_new)")
print()