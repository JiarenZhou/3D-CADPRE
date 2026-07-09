import warnings
warnings.filterwarnings("ignore")

import os
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# STYLE
# =====================================================
plt.rcParams.update({

    "font.family": "Arial",

    "axes.unicode_minus": False,

    # ------------------------
    # Global Font Size
    # ------------------------
    "font.size": 18,

    "axes.titlesize": 22,

    "axes.labelsize": 20,

    "xtick.labelsize": 18,

    "ytick.labelsize": 18,

    "legend.fontsize": 18,

    "figure.titlesize": 24

})

sns.set_style("whitegrid")

# =====================================================
# RANDOM SEED
# =====================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# PATH
# =====================================================
data_path = r"C:/Users/Jeron/Desktop/code/反演/feature/selected_features.xlsx"
fig_dir = r"C:/Users/Jeron/Desktop/code/反演/outputs/figures"
metric_dir = r"C:/Users/Jeron/Desktop/code/反演/outputs/metrics"

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(data_path)
y = df["Pn"].values.astype(np.float32)
all_features = df.drop(columns=["Pn"]).columns.tolist()

# =====================================================
# METRIC
# =====================================================
def calc_metric(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = rmse / np.mean(y_true) * 100
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, rrmse, mae

# =====================================================
# CV
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# =====================================================
# BPNN
# =====================================================
class ImprovedBPNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# =====================================================
# WRAPPER
# =====================================================
class BPNNWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim,
        lr=0.001,
        epochs=300,
        batch_size=32,
        patience=30
    ):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(
            y.reshape(-1, 1),
            dtype=torch.float32
        )
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model = ImprovedBPNN(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        best_loss = np.inf
        patience_counter = 0
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break
        self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(
                X,
                dtype=torch.float32
            )
            pred = self.model(X_tensor)
        return pred.numpy().flatten()

# =====================================================
# STACKING FUNCTION
# =====================================================
def run_stacking(X, y):

    # =================================================
    # BEST PARAMS
    # =================================================
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=SEED
    )

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=SEED,
        verbosity=0
    )

    bpnn = BPNNWrapper(
        input_dim=X.shape[1],
        lr=0.001,
        patience=30
    )

    base_models = {
        "RF": rf,
        "XGB": xgb,
        "BPNN": bpnn
    }

    # =================================================
    # OOF
    # =================================================
    oof_predictions = np.zeros(
        (X.shape[0], len(base_models))
    )

    for model_idx, (name, model) in enumerate(base_models.items()):
        oof_pred = np.zeros(len(y))
        for train_idx, valid_idx in kf.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_valid = X[valid_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            oof_pred[valid_idx] = pred
        oof_predictions[:, model_idx] = oof_pred

    # =================================================
    # META MODEL
    # =================================================
    meta_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=SEED
    )

    meta_model.fit(oof_predictions, y)
    stack_pred = meta_model.predict(oof_predictions)
    return calc_metric(y, stack_pred)

# =====================================================
# BASELINE
# =====================================================
print("\n====================================")
print("BASELINE STACKING")
print("====================================")
X_full = df[all_features].values.astype(np.float32)
baseline_metric = run_stacking(X_full, y)
baseline_r2 = baseline_metric[0]
print(f"\nBaseline R2 = {baseline_r2:.6f}")

# =====================================================
# FEATURE REMOVAL ANALYSIS
# =====================================================
results = []

print("\n====================================")
print("LOFO FEATURE IMPORTANCE")
print("====================================")

for feature in all_features:
    print(f"\nRemoving Feature: {feature}")
    remain_features = [
        f for f in all_features
        if f != feature
    ]
    X_new = df[remain_features].values.astype(np.float32)
    metric = run_stacking(X_new, y)
    new_r2 = metric[0]
    r2_drop = baseline_r2 - new_r2
    results.append({
        "Removed_Feature": feature,
        "R2_After_Removal": new_r2,
        "R2_Drop": r2_drop,
        "RMSE": metric[1],
        "rRMSE": metric[2],
        "MAE": metric[3]
    })
    print(f"R2 = {new_r2:.6f}")
    print(f"R2 Drop = {r2_drop:.6f}")

# =====================================================
# SAVE CSV
# =====================================================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(
    by="R2_Drop",
    ascending=False
)

csv_path = os.path.join(
    metric_dir,
    "stacking_feature_importance_lofo.csv"
)

result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# FIGURE 1
# =====================================================
plt.figure(figsize=(12, 6))

sns.barplot(
    data=result_df,
    x="Removed_Feature",
    y="R2_Drop"
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("R2 Drop")
plt.title("Feature Importance (LOFO Stacking)")
plt.tight_layout()

fig1_path = os.path.join(
    fig_dir,
    "stacking_feature_importance.png"
)

plt.savefig(
    fig1_path,
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 2
# =====================================================
# =====================================================
# FEATURE IMPACT HEATMAPS
# =====================================================
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 10)
)

metrics = [
    ("R2_Drop", "R² Drop", "Reds"),
    ("RMSE", "RMSE", "Blues"),
    ("MAE", "MAE", "Greens")
]

for ax, (metric_col, title, cmap) in zip(axes, metrics):

    heatmap_data = pd.DataFrame(
        result_df[metric_col].values,
        index=result_df["Removed_Feature"],
        columns=[title]
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title)

    ax.set_xlabel("")
    ax.set_ylabel("")

plt.suptitle(
    "Feature Removal Impact on Stacking Performance",
    fontsize=16
)

plt.tight_layout()

fig2_path = os.path.join(
    fig_dir,
    "stacking_feature_heatmap.png"
)

plt.savefig(
    fig2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# DONE
# =====================================================
print("\n====================================")
print("ANALYSIS COMPLETED")
print("====================================")

print(f"\nBaseline R2: {baseline_r2:.6f}")

print("\nSaved CSV:")
print(csv_path)

print("\nSaved Figures:")
print(fig1_path)
print(fig2_path)
