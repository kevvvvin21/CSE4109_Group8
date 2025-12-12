#!/usr/bin/env python3

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb

# ------------------------------
# Graph building from features
# ------------------------------


def build_edge_index_knn(
    X: Tensor,
    k: int = 10,
    metric: str = "cosine",
    mutual: bool = True,
    add_loops: bool = True,
) -> Tensor:
    """Build edge_index by connecting each node to its k nearest neighbors.

    Args:
        X: (N, F) float tensor. For metric='cosine', rows should be L2-normalized.
        k: number of neighbors per node (excluding self)
        metric: 'cosine' or 'euclidean'
        mutual: if True, keep i<->j only when both are in each other's top-k
        add_loops: if True, add self-loops to edge_index
    Returns:
        edge_index: (2, E) long tensor
    """
    X = X.to(torch.float32)
    N = X.size(0)

    if metric == "cosine":
        Xn = X / (X.norm(dim=1, keepdim=True).clamp(min=1e-8))
        S = Xn @ Xn.t()  # (N,N) cosine similarity
        vals, idx = torch.topk(S, k=k + 1, dim=1)  # +1 for self
        idx = idx[:, 1:]  # drop self
    elif metric == "euclidean":
        # Compute squared distances via (x - y)^2 = ||x||^2 + ||y||^2 - 2 x y
        X2 = (X * X).sum(dim=1, keepdim=True)  # (N,1)
        D2 = X2 + X2.t() - 2.0 * (X @ X.t())  # (N,N)
        # smaller is closer → take smallest distances
        vals, idx = torch.topk(-D2, k=k + 1, dim=1)
        idx = idx[:, 1:]  # drop self
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    row = torch.arange(N).unsqueeze(1).repeat(1, idx.size(1)).reshape(-1)
    col = idx.reshape(-1)
    edges = torch.stack([row, col], dim=0)  # (2, N*k)

    if mutual:
        # Keep mutual neighbors using a boolean adjacency
        A = torch.zeros((N, N), dtype=torch.bool)
        A[edges[0], edges[1]] = True
        mutual_mask = A & A.t()
        i, j = torch.nonzero(mutual_mask, as_tuple=True)
        edges = torch.stack([i, j], dim=0)

    edge_index = edges.long()

    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    return edge_index


# ------------------------------
# CSV loading
# ------------------------------


def load_csv_features(
    csv_path: str,
    label_col: Optional[str] = None,
    drop_cols: Optional[str] = None,
) -> Tuple[Tensor, Optional[Tensor], List[str]]:
    df = pd.read_csv(csv_path)

    cols_to_drop = []
    y = None

    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in CSV columns.")
        y = torch.tensor(df[label_col].values, dtype=torch.long)
        cols_to_drop.append(label_col)

    if drop_cols is not None:
        for c in drop_cols.split(","):
            c = c.strip()
            if c and c in df.columns and c not in cols_to_drop:
                cols_to_drop.append(c)

    X_df = df.drop(columns=cols_to_drop) if cols_to_drop else df
    cols = list(X_df.columns)
    X = torch.tensor(X_df.values, dtype=torch.float32)
    return X, y, cols


# ------------------------------
# GAT Embedding Model (No classifier head)
# ------------------------------


class GAT_Embedding(nn.Module):
    """GAT layers for generating node embeddings (no classification head)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_gat_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.3,
        attn_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_gat_layers = num_gat_layers
        self.feat_dropout = dropout
        self.attn_dropout = attn_dropout
        self.heads = heads
        self.hidden = hidden

        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_dim,
                hidden,
                heads=heads,
                concat=True,
                dropout=attn_dropout,
                add_self_loops=False,
            )
        )
        self.bn_layers.append(nn.BatchNorm1d(hidden * heads))
        self.skip_layers.append(nn.Linear(in_dim, hidden * heads, bias=False))

        # Middle GAT layers
        for _ in range(num_gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden * heads,
                    hidden,
                    heads=heads,
                    concat=True,
                    dropout=attn_dropout,
                    add_self_loops=False,
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden * heads))
            self.skip_layers.append(
                nn.Linear(hidden * heads, hidden * heads, bias=False)
            )

        # Last GAT layer (concat=False to get single hidden dimension)
        if num_gat_layers > 1:
            self.gat_layers.append(
                GATConv(
                    hidden * heads,
                    hidden,
                    heads=1,
                    concat=False,
                    dropout=attn_dropout,
                    add_self_loops=False,
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden))
            self.skip_layers.append(nn.Linear(hidden * heads, hidden, bias=False))

    def forward(self, x: Tensor, edge_index: Tensor, return_all_layers: bool = False):
        """Forward pass returns node embeddings.

        Args:
            x: Input features
            edge_index: Graph edges
            return_all_layers: If True, return embeddings from all layers concatenated

        Returns:
            If return_all_layers=False: final layer embeddings (N, hidden)
            If return_all_layers=True: all layer embeddings concatenated
        """
        h = F.dropout(x, p=self.feat_dropout * 0.5, training=self.training)

        all_embeddings = []

        for i in range(self.num_gat_layers):
            h_in = h
            h = self.gat_layers[i](h, edge_index)
            h = self.bn_layers[i](h)
            h = h + self.skip_layers[i](h_in)  # Skip connection
            h = F.elu(h)
            h = F.dropout(h, p=self.feat_dropout, training=self.training)

            if return_all_layers:
                all_embeddings.append(h)

        if return_all_layers and len(all_embeddings) > 1:
            # Concatenate embeddings from all layers
            return torch.cat(all_embeddings, dim=-1)
        return h  # Return final embeddings (N, hidden)


# ------------------------------
# Train / Eval
# ------------------------------


def split_indices(
    N: int, train=0.6, val=0.2, seed: int = 42, y: Optional[Tensor] = None
):
    """Split indices with optional stratification by labels."""
    if y is None:
        # Original random splitting
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        n_tr = int(N * train)
        n_va = int(N * val)
        train_idx = perm[:n_tr]
        val_idx = perm[n_tr : n_tr + n_va]
        test_idx = perm[n_tr + n_va :]
        return train_idx, val_idx, test_idx

    # Stratified splitting - maintain class proportions
    np.random.seed(seed)

    indices = np.arange(N)
    y_np = y.numpy()

    # Get indices for each class
    unique_labels = np.unique(y_np)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in unique_labels:
        # Get all indices for this class
        class_indices = indices[y_np == label]
        n_class = len(class_indices)

        # Shuffle indices for this class
        np.random.shuffle(class_indices)

        # Split proportionally
        n_tr = int(n_class * train)
        n_va = int(n_class * val)

        train_indices.extend(class_indices[:n_tr])
        val_indices.extend(class_indices[n_tr : n_tr + n_va])
        test_indices.extend(class_indices[n_tr + n_va :])

    # Shuffle each split to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Convert back to torch tensors
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)

    return train_idx, val_idx, test_idx


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for embedding classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.first_forward = True  # Flag to print dimensions on first forward pass

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.fc_out = nn.Linear(prev_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        h = x

        if self.first_forward:
            print(f"\n[MLP Forward Pass] Dimension trace:")
            print(f"  Input: {h.shape}")

        # Hidden layers with residual connections
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(h)

            # Batch normalization
            if self.use_batchnorm:
                h = self.bns[i](h)

            # Residual connection (if dimensions match)
            if self.use_residual and h.shape == h_in.shape:
                h = h + h_in
                if self.first_forward:
                    print(
                        f"  Hidden Layer {i+1}: {h_in.shape} → {h.shape} (with residual)"
                    )
            else:
                if self.first_forward:
                    print(f"  Hidden Layer {i+1}: {h_in.shape} → {h.shape}")

            # Activation and dropout
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Output layer (no activation)
        out = self.fc_out(h)

        if self.first_forward:
            print(f"  Output Layer: {h.shape} → {out.shape}")
            self.first_forward = False  # Only print once

        return out


@dataclass
class CFG:
    mode: str = "csv"
    csv_path: Optional[str] = None
    label_col: Optional[str] = None
    drop_cols: Optional[str] = None
    hidden: int = 64
    num_gat_layers: int = 2  # Number of GAT layers before XGBoost
    heads: int = 8
    dropout: float = 0.3
    attn_dropout: float = 0.5
    k: int = 10
    metric: str = "cosine"  # cosine | euclidean
    mutual: bool = True
    gnn_epochs: int = 100  # Epochs for training GAT embeddings
    lr: float = 0.01
    weight_decay: float = 5e-4
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    # Feature combination options
    use_all_layers: bool = True  # Use embeddings from all GAT layers
    concat_original: bool = True  # Concatenate original features with embeddings
    # MLP Classifier parameters (for embedding training)
    mlp_hidden_dims: List[int] = None  # e.g., [128, 64] for 2 hidden layers
    mlp_dropout: float = 0.3
    mlp_use_batchnorm: bool = True
    mlp_use_residual: bool = True
    # XGBoost parameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: int = 1
    xgb_gamma: float = 0.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0


def train_gat_embeddings(
    model: nn.Module,
    data: Data,
    cfg: CFG,
    device: torch.device,
) -> nn.Module:
    """Train GAT model to learn good embeddings using supervised loss."""

    # Add a temporary classifier head for training
    num_classes = int(data.y.max().item() + 1)

    # Determine embedding dimension based on config
    if cfg.use_all_layers and cfg.num_gat_layers > 1:
        # If using all layers, dimensions will be concatenated
        # All but last layer: hidden * heads, last layer: hidden
        embed_dim = cfg.hidden * cfg.heads * (cfg.num_gat_layers - 1) + cfg.hidden
    else:
        embed_dim = cfg.hidden

    # Create MLP classifier with configurable hidden layers
    mlp_hidden_dims = cfg.mlp_hidden_dims if cfg.mlp_hidden_dims else []
    temp_classifier = MLPClassifier(
        input_dim=embed_dim,
        hidden_dims=mlp_hidden_dims,
        output_dim=num_classes,
        dropout=cfg.mlp_dropout,
        use_batchnorm=cfg.mlp_use_batchnorm,
        use_residual=cfg.mlp_use_residual,
    ).to(device)

    print(f"MLP Classifier architecture:")
    print(f"  Input dim: {embed_dim}")
    if mlp_hidden_dims:
        print(f"  Hidden layers: {mlp_hidden_dims}")
    else:
        print(f"  Hidden layers: None (direct classification)")
    print(f"  Output dim: {num_classes}")
    print(f"  Dropout: {cfg.mlp_dropout}")
    print(f"  BatchNorm: {cfg.mlp_use_batchnorm}")
    print(f"  Residual: {cfg.mlp_use_residual}")

    opt = torch.optim.Adam(
        list(model.parameters()) + list(temp_classifier.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    max_patience = 30

    print("\n" + "=" * 50)
    print("Phase 1: Training GAT for embeddings")
    print("=" * 50)
    print(f"Using all layers: {cfg.use_all_layers}")
    print(f"Embedding dimension: {embed_dim}")

    for epoch in range(1, cfg.gnn_epochs + 1):
        model.train()
        temp_classifier.train()
        opt.zero_grad(set_to_none=True)

        # Get embeddings
        embeddings = model(
            data.x, data.edge_index, return_all_layers=cfg.use_all_layers
        )

        # Print embedding dimensions on first epoch
        if epoch == 1:
            print(f"\n[Epoch 1] GAT output embedding shape: {embeddings.shape}")
            print(f"  - Number of nodes: {embeddings.shape[0]}")
            print(f"  - Embedding dimension: {embeddings.shape[1]}")

        # Classify
        out = temp_classifier(embeddings)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        # Validation
        model.eval()
        temp_classifier.eval()
        with torch.no_grad():
            embeddings = model(
                data.x, data.edge_index, return_all_layers=cfg.use_all_layers
            )
            out = temp_classifier(embeddings)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])

            train_preds = out[data.train_mask].argmax(dim=1)
            val_preds = out[data.val_mask].argmax(dim=1)
            train_acc = (train_preds == data.y[data.train_mask]).float().mean()
            val_acc = (val_preds == data.y[data.val_mask]).float().mean()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if epoch % max(1, cfg.gnn_epochs // 10) == 0 or epoch in (1, 2, 5, 10):
            print(
                f"Epoch {epoch:4d} | loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | "
                f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f}"
            )

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_loss: {best_val_loss:.4f}")

    return model


def train_xgboost_on_embeddings(
    model: nn.Module,
    data: Data,
    cfg: CFG,
    device: torch.device,
):
    """Extract embeddings and train XGBoost classifier."""

    print("\n" + "=" * 50)
    print("Phase 2: Training XGBoost on GAT embeddings")
    print("=" * 50)
    print(f"Concatenate original features: {cfg.concat_original}")

    model.eval()
    with torch.no_grad():
        # Get embeddings for all nodes
        embeddings = model(
            data.x, data.edge_index, return_all_layers=cfg.use_all_layers
        )
        embeddings_np = embeddings.cpu().numpy()

    # Optionally concatenate original features with embeddings
    if cfg.concat_original:
        original_features_np = data.x.cpu().numpy()
        combined_features = np.concatenate(
            [original_features_np, embeddings_np], axis=1
        )
        print(
            f"Combined features: {original_features_np.shape} (orig) + {embeddings_np.shape} (embed) = {combined_features.shape}"
        )
    else:
        combined_features = embeddings_np
        print(f"Using only embeddings: {embeddings_np.shape}")

    # Prepare data for XGBoost
    y_np = data.y.cpu().numpy()

    train_mask_np = data.train_mask.cpu().numpy()
    val_mask_np = data.val_mask.cpu().numpy()
    test_mask_np = data.test_mask.cpu().numpy()

    X_train = combined_features[train_mask_np]
    y_train = y_np[train_mask_np]

    X_val = combined_features[val_mask_np]
    y_val = y_np[val_mask_np]

    X_test = combined_features[test_mask_np]
    y_test = y_np[test_mask_np]

    print(f"Training XGBoost with:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Train XGBoost
    num_classes = int(data.y.max().item() + 1)

    if num_classes == 2:
        # Binary classification
        xgb_model = xgb.XGBClassifier(
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            n_estimators=cfg.xgb_n_estimators,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            min_child_weight=cfg.xgb_min_child_weight,
            gamma=cfg.xgb_gamma,
            reg_alpha=cfg.xgb_reg_alpha,
            reg_lambda=cfg.xgb_reg_lambda,
            random_state=cfg.seed,
            eval_metric="logloss",
            early_stopping_rounds=30,
            tree_method="hist",
            n_jobs=-1,
        )
    else:
        # Multi-class classification
        xgb_model = xgb.XGBClassifier(
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            n_estimators=cfg.xgb_n_estimators,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            min_child_weight=cfg.xgb_min_child_weight,
            gamma=cfg.xgb_gamma,
            reg_alpha=cfg.xgb_reg_alpha,
            reg_lambda=cfg.xgb_reg_lambda,
            random_state=cfg.seed,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            tree_method="hist",
            objective="multi:softmax",
            n_jobs=-1,
        )

    # Fit with validation set for early stopping
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Evaluate on all splits
    print("\n" + "=" * 50)
    print("Final Results")
    print("=" * 50)

    train_pred = xgb_model.predict(X_train)
    val_pred = xgb_model.predict(X_val)
    test_pred = xgb_model.predict(X_test)

    # Get probability predictions for AUROC
    train_pred_proba = xgb_model.predict_proba(X_train)
    val_pred_proba = xgb_model.predict_proba(X_val)
    test_pred_proba = xgb_model.predict_proba(X_test)

    # Calculate metrics for test set
    test_acc = accuracy_score(y_test, test_pred)
    test_precision = precision_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_recall = recall_score(y_test, test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    # Calculate AUROC
    if num_classes == 2:
        # Binary classification - use probability of positive class
        test_auroc = roc_auc_score(y_test, test_pred_proba[:, 1])
    else:
        # Multi-class classification - use one-vs-rest
        test_auroc = roc_auc_score(
            y_test, test_pred_proba, multi_class="ovr", average="weighted"
        )

    print(f"\n{'='*60}")
    print(f"{'TEST SET METRICS':^60}")
    print(f"{'='*60}")
    print(f"  AUROC:      {test_auroc:.4f}")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  Precision:  {test_precision:.4f}")
    print(f"  Recall:     {test_recall:.4f}")
    print(f"  F1:         {test_f1:.4f}")
    print(f"{'='*60}")

    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred))

    print(f"\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Feature importance from XGBoost
    print(f"\nTop 10 Most Important Feature Dimensions:")
    importance = xgb_model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature dim {idx}: {importance[idx]:.4f}")

    # Return all metrics
    metrics = {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
    }
    return xgb_model, metrics


def train_xgboost_baseline(
    data: Data,
    cfg: CFG,
):
    """Train XGBoost on original features only (no GAT) as a baseline."""

    print("\n" + "=" * 50)
    print("Baseline: Training XGBoost on original features only")
    print("=" * 50)

    # Prepare data
    X_np = data.x.cpu().numpy()
    y_np = data.y.cpu().numpy()

    train_mask_np = data.train_mask.cpu().numpy()
    val_mask_np = data.val_mask.cpu().numpy()
    test_mask_np = data.test_mask.cpu().numpy()

    X_train = X_np[train_mask_np]
    y_train = y_np[train_mask_np]

    X_val = X_np[val_mask_np]
    y_val = y_np[val_mask_np]

    X_test = X_np[test_mask_np]
    y_test = y_np[test_mask_np]

    print(f"Training baseline XGBoost with:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Train XGBoost
    num_classes = int(data.y.max().item() + 1)

    xgb_model = xgb.XGBClassifier(
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        n_estimators=cfg.xgb_n_estimators,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        min_child_weight=cfg.xgb_min_child_weight,
        gamma=cfg.xgb_gamma,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        random_state=cfg.seed,
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    test_pred = xgb_model.predict(X_test)
    test_pred_proba = xgb_model.predict_proba(X_test)

    # Calculate all metrics
    test_acc = accuracy_score(y_test, test_pred)
    test_precision = precision_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_recall = recall_score(y_test, test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    # Calculate AUROC
    test_auroc = roc_auc_score(y_test, test_pred_proba[:, 1])

    print(f"\n{'='*60}")
    print(f"{'BASELINE TEST SET METRICS':^60}")
    print(f"{'='*60}")
    print(f"  AUROC:      {test_auroc:.4f}")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  Precision:  {test_precision:.4f}")
    print(f"  Recall:     {test_recall:.4f}")
    print(f"  F1:         {test_f1:.4f}")
    print(f"{'='*60}")

    # Return all metrics
    metrics = {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
    }
    return metrics


def run(cfg: CFG):
    # Setup device
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    print(f"Using device: {device}")
    print(f"Config: {cfg}")

    # Load features + labels
    X_raw, y, cols = load_csv_features(cfg.csv_path, cfg.label_col, cfg.drop_cols)
    print(f"\n#samples: {len(y)}")
    print(f"#positive labels (sum): {int(y.sum().item())}")
    print(f"#features: {X_raw.shape[1]}")

    num_classes = int(y.max().item() + 1) if y is not None else 0
    print(f"#classes: {num_classes}")

    # Standardize for Euclidean; L2-normalize for cosine
    X = X_raw.clone()
    if cfg.metric == "euclidean":
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X.numpy())
        X = torch.tensor(X_np, dtype=torch.float32)
    else:  # cosine
        X = X / (X.norm(dim=1, keepdim=True).clamp(min=1e-8))

    # Build edges from features
    print(
        f"\nBuilding kNN graph with k={cfg.k}, metric={cfg.metric}, mutual={cfg.mutual}"
    )
    edge_index = build_edge_index_knn(
        X, k=cfg.k, metric=cfg.metric, mutual=cfg.mutual, add_loops=True
    )
    print(f"Graph edges: {edge_index.shape[1]}")

    # For node features, we can use the standardized/normalized features.
    data = Data(x=X, edge_index=edge_index)
    if y is not None:
        data.y = y

    # Train/val/test split masks
    if y is not None:
        N = data.num_nodes
        tr_idx, va_idx, te_idx = split_indices(N, seed=cfg.seed, y=y)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[tr_idx] = True
        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[va_idx] = True
        test_mask = torch.zeros(N, dtype=torch.bool)
        test_mask[te_idx] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    # Move data to device
    data = data.to(device)

    # Print class distribution for verification
    if y is not None:
        print(f"\nClass distribution in dataset: {torch.bincount(y)}")
        print(
            f"Train: {torch.bincount(y[tr_idx])}, "
            f"Val: {torch.bincount(y[va_idx])}, "
            f"Test: {torch.bincount(y[te_idx])}"
        )

    # Create GAT embedding model
    in_dim = data.num_features
    model = GAT_Embedding(
        in_dim,
        cfg.hidden,
        num_gat_layers=cfg.num_gat_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
    )
    model = model.to(device)
    print(f"\nGAT model has {sum(p.numel() for p in model.parameters())} parameters")

    # Baseline: XGBoost on original features only
    baseline_metrics = train_xgboost_baseline(data, cfg)

    # Phase 1: Train GAT to learn embeddings
    model = train_gat_embeddings(model, data, cfg, device)

    # Phase 2: Train XGBoost on the learned embeddings
    xgb_model, gat_xgb_metrics = train_xgboost_on_embeddings(model, data, cfg, device)

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - ALL METRICS")
    print("=" * 80)
    print(f"{'Metric':<15} {'Baseline':<15} {'GAT+XGBoost':<15} {'Improvement':<15}")
    print("-" * 80)

    for metric_name in ["auroc", "accuracy", "precision", "recall", "f1"]:
        baseline_val = baseline_metrics[metric_name]
        gat_val = gat_xgb_metrics[metric_name]
        improvement = gat_val - baseline_val
        print(
            f"{metric_name.upper():<15} {baseline_val:<15.4f} {gat_val:<15.4f} {improvement:+.4f} ({improvement*100:+.2f}%)"
        )

    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - GAT layers: {cfg.num_gat_layers}")
    print(f"  - Hidden dim: {cfg.hidden}")
    print(f"  - Use all layers: {cfg.use_all_layers}")
    print(f"  - Concat original: {cfg.concat_original}")
    print(f"  - k-NN: k={cfg.k}, metric={cfg.metric}, mutual={cfg.mutual}")

    return model, xgb_model


# ------------------------------
# CLI
# ------------------------------


def parse_args() -> CFG:
    p = argparse.ArgumentParser(
        description="GAT + XGBoost: Use GAT layers for embeddings, then XGBoost for classification"
    )
    p.add_argument("--mode", type=str, default="csv", choices=["csv"])
    p.add_argument("--csv_path", type=str, default="balanced_data_new.csv")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument(
        "--drop_cols",
        type=str,
        default=None,
        help="comma-separated columns to drop (e.g., id,index)",
    )
    # GAT parameters
    p.add_argument("--hidden", type=int, default=32, help="Hidden dimension size")
    p.add_argument("--num_gat_layers", type=int, default=3, help="Number of GAT layers")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.4)
    # Graph construction
    p.add_argument("--k", type=int, default=10, help="Number of neighbors in kNN graph")
    p.add_argument(
        "--metric", type=str, default="euclidean", choices=["cosine", "euclidean"]
    )
    p.add_argument("--mutual", action="store_true", help="keep only mutual kNN edges")
    # Training parameters
    p.add_argument("--gnn_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    # Feature combination options
    p.add_argument(
        "--use_all_layers",
        action="store_true",
        default=True,
        help="Use embeddings from all GAT layers (concatenated)",
    )
    p.add_argument(
        "--no_use_all_layers",
        action="store_false",
        dest="use_all_layers",
        help="Use only final layer embeddings",
    )
    p.add_argument(
        "--concat_original",
        action="store_true",
        default=False,
        help="Concatenate original features with GAT embeddings",
    )
    p.add_argument(
        "--no_concat_original",
        action="store_false",
        dest="concat_original",
        help="Use only GAT embeddings (no original features)",
    )
    # MLP Classifier parameters (for embedding training)
    p.add_argument(
        "--mlp_hidden_dims",
        type=str,
        default="128,64,16",
        help="Comma-separated hidden layer dimensions for MLP (e.g., '128,64' for 2 layers). Leave empty for direct classification.",
    )
    p.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for MLP classifier",
    )
    p.add_argument(
        "--mlp_use_batchnorm",
        action="store_true",
        default=True,
        help="Use batch normalization in MLP",
    )
    p.add_argument(
        "--no_mlp_batchnorm",
        action="store_false",
        dest="mlp_use_batchnorm",
        help="Disable batch normalization in MLP",
    )
    p.add_argument(
        "--mlp_use_residual",
        action="store_true",
        default=True,
        help="Use residual connections in MLP (when dimensions match)",
    )
    p.add_argument(
        "--no_mlp_residual",
        action="store_false",
        dest="mlp_use_residual",
        help="Disable residual connections in MLP",
    )
    # XGBoost parameters
    p.add_argument("--xgb_max_depth", type=int, default=5)
    p.add_argument("--xgb_learning_rate", type=float, default=0.1)
    p.add_argument("--xgb_n_estimators", type=int, default=300)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    p.add_argument("--xgb_min_child_weight", type=int, default=1)
    p.add_argument("--xgb_gamma", type=float, default=0.0)
    p.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    a = p.parse_args()  # Use None instead of [] for command line args

    # Parse mlp_hidden_dims from string to list
    args_dict = vars(a)
    if args_dict.get("mlp_hidden_dims") is not None:
        try:
            args_dict["mlp_hidden_dims"] = [
                int(x.strip()) for x in args_dict["mlp_hidden_dims"].split(",")
            ]
        except:
            args_dict["mlp_hidden_dims"] = None

    return CFG(**args_dict)


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
