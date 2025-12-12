"""
Random Forest Model Training Script
Train and evaluate a Random Forest classifier

Usage:
    python train_random_forest_standalone.py

Requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RANDOM FOREST MODEL TRAINING")
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
# 2. TRAIN RANDOM FOREST MODEL
# =============================================================================
print("="*70)
print("Step 2: Training Random Forest Model")
print("="*70)

start_time = datetime.now()

# Random Forest configuration
rf_model = RandomForestClassifier(
    n_estimators=500,           # Number of trees in the forest
    max_depth=15,               # Maximum depth of each tree
    min_samples_split=10,       # Minimum samples required to split a node
    min_samples_leaf=4,         # Minimum samples required at a leaf node
    max_features='sqrt',        # Number of features to consider for best split
    bootstrap=True,             # Use bootstrap samples
    oob_score=True,            # Use out-of-bag samples for validation
    n_jobs=-1,                 # Use all available CPU cores
    random_state=42,           # For reproducibility
    verbose=1                  # Show progress
)

print("\nTraining in progress...")
print("-" * 70)
rf_model.fit(X_train, y_train)
print("-" * 70)

training_time = (datetime.now() - start_time).total_seconds()

print(f"\n✓ Training complete!")
print(f"  Training time: {training_time:.1f} seconds")
print(f"  Number of trees: {rf_model.n_estimators}")
print(f"  Out-of-Bag Score: {rf_model.oob_score_:.4f}")
print(f"  Number of features used: {rf_model.n_features_in_}\n")

# =============================================================================
# 3. MAKE PREDICTIONS
# =============================================================================
print("="*70)
print("Step 3: Making predictions on all datasets")
print("="*70)

# Predictions on training set
y_train_pred = rf_model.predict(X_train)
y_train_pred_proba = rf_model.predict_proba(X_train)[:, 1]

# Predictions on validation set
y_val_pred = rf_model.predict(X_val)
y_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]

# Predictions on test set
y_test_pred = rf_model.predict(X_test)
y_test_pred_proba = rf_model.predict_proba(X_test)[:, 1]

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
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print("-" * 70)
print(feature_importance.head(15).to_string(index=False))
print()

# =============================================================================
# 6. SAVE MODEL
# =============================================================================
print("="*70)
print("Step 6: Saving model and predictions")
print("="*70)

# Save model
model_path = 'random_forest_model.pkl'
joblib.dump(rf_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'probability': y_test_pred_proba
})
predictions_path = 'rf_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✓ Predictions saved: {predictions_path}")

# Save feature importance
importance_path = 'feature_importance.csv'
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
axes[0, 0].set_xlabel('Importance (Gini)', fontsize=11)
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
axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
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
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')

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

# 6. Tree Depth Distribution (sample of trees)
tree_depths = [tree.get_depth() for tree in rf_model.estimators_[:100]]
axes[1, 2].hist(tree_depths, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 2].set_xlabel('Tree Depth', fontsize=11)
axes[1, 2].set_ylabel('Frequency', fontsize=11)
axes[1, 2].set_title('Distribution of Tree Depths (First 100 Trees)', fontsize=13, fontweight='bold')
axes[1, 2].axvline(x=np.mean(tree_depths), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(tree_depths):.1f}')
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.suptitle('Random Forest Model - Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

viz_path = 'random_forest_results.png'
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

print(f"\nModel Performance on Test Set:")
print(f"  • Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • AUC-ROC:   {test_auc:.4f}")
print(f"  • Precision (Class 0): {classification_report(y_test, y_test_pred, output_dict=True)['0']['precision']:.4f}")
print(f"  • Precision (Class 1): {classification_report(y_test, y_test_pred, output_dict=True)['1']['precision']:.4f}")
print(f"  • Recall (Class 0):    {classification_report(y_test, y_test_pred, output_dict=True)['0']['recall']:.4f}")
print(f"  • Recall (Class 1):    {classification_report(y_test, y_test_pred, output_dict=True)['1']['recall']:.4f}")

print(f"\nModel Information:")
print(f"  • Number of trees: {rf_model.n_estimators}")
print(f"  • Training time: {training_time:.1f} seconds")
print(f"  • Out-of-Bag Score: {rf_model.oob_score_:.4f}")
print(f"  • Number of features: {rf_model.n_features_in_}")

print(f"\nTop 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\nFiles Generated:")
print(f"  1. {model_path} - Trained Random Forest model")
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
print("  model = joblib.load('random_forest_model.pkl')")
print("  predictions = model.predict(X_new)")
print("  probabilities = model.predict_proba(X_new)")
print()