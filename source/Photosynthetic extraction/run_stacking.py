import warnings
warnings.filterwarnings("ignore")

import os
import time
import random

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

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
# GLOBAL STYLE
# =====================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

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
model_dir = r"C:/Users/Jeron/Desktop/code/反演/outputs/models"

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(data_path)

y = df["Pn"].values.astype(np.float32)

X = df.drop(columns=["Pn"]).values.astype(np.float32)

feature_names = df.drop(columns=["Pn"]).columns.tolist()

# =====================================================
# METRIC FUNCTION
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
# =====================================================
# BPNN MODEL
# =====================================================
# =====================================================
class ImprovedBPNN(nn.Module):

    def __init__(self, input_dim, hidden1, hidden2):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden1),

            nn.ReLU(),

            nn.Linear(hidden1, hidden2),

            nn.ReLU(),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x):

        return self.net(x)

# =====================================================
# PYTORCH WRAPPER
# =====================================================
class BPNNWrapper(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        input_dim,
        hidden1=128,
        hidden2=64,
        lr=0.001,
        epochs=300,
        batch_size=32,
        patience=30,
        verbose=False
    ):

        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        self.model = None

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

        self.model = ImprovedBPNN(
            self.input_dim,
            self.hidden1,
            self.hidden2
        )

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        best_loss = np.inf
        patience_counter = 0

        self.model.train()

        for epoch in range(self.epochs):

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
# BASE MODELS
# =====================================================
rf = RandomForestRegressor(
    random_state=SEED
)

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=SEED,
    verbosity=0
)

bpnn = BPNNWrapper(
    input_dim=X.shape[1]
)

# =====================================================
# PARAM GRID
# =====================================================
rf_param = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

xgb_param = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3]
}

bpnn_param = {
    "hidden1": [64, 128],
    "hidden2": [32, 64],
    "lr": [0.001, 0.01],
    "patience": [20, 30]
}

# =====================================================
# GRID SEARCH
# =====================================================
print("\n====================================")
print("HYPERPARAMETER SEARCH")
print("====================================")

# =====================================================
# RF
# =====================================================
print("\nSearching RF...")

rf_grid = GridSearchCV(
    rf,
    rf_param,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

rf_grid.fit(X, y)

best_rf = rf_grid.best_estimator_

print("RF Best Params:")
print(rf_grid.best_params_)

# =====================================================
# XGB
# =====================================================
print("\nSearching XGB...")

xgb_grid = GridSearchCV(
    xgb,
    xgb_param,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

xgb_grid.fit(X, y)

best_xgb = xgb_grid.best_estimator_

print("XGB Best Params:")
print(xgb_grid.best_params_)

# =====================================================
# BPNN
# =====================================================
print("\nSearching BPNN...")

bpnn_grid = GridSearchCV(
    bpnn,
    bpnn_param,
    cv=kf,
    scoring="r2",
    n_jobs=1
)

bpnn_grid.fit(X, y)

best_bpnn = bpnn_grid.best_estimator_

print("BPNN Best Params:")
print(bpnn_grid.best_params_)

# =====================================================
# BASE MODELS
# =====================================================
base_models = {
    "RF": best_rf,
    "XGB": best_xgb,
    "BPNN": best_bpnn
}

# =====================================================
# =====================================================
# OOF STACKING
# =====================================================
# =====================================================
print("\n====================================")
print("STACKING OOF TRAINING")
print("====================================")

# =====================================================
# OOF MATRIX
# =====================================================
oof_predictions = np.zeros(
    (X.shape[0], len(base_models))
)

base_test_predictions = {}

# =====================================================
# OOF LOOP
# =====================================================
for model_idx, (model_name, model) in enumerate(base_models.items()):

    print(f"\nOOF -> {model_name}")

    oof_pred = np.zeros(len(y))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

        print(f"Fold {fold + 1}")

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_valid = X[valid_idx]

        model.fit(X_train, y_train)

        pred_valid = model.predict(X_valid)

        oof_pred[valid_idx] = pred_valid

    oof_predictions[:, model_idx] = oof_pred

    base_test_predictions[model_name] = oof_pred

# =====================================================
# META LEARNER
# =====================================================
meta_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=SEED
)

meta_model.fit(oof_predictions, y)

# =====================================================
# FINAL PREDICTION
# =====================================================
stack_pred = meta_model.predict(oof_predictions)

# =====================================================
# METRICS
# =====================================================
m_stack = calc_metric(y, stack_pred)

print("\n====================================")
print("STACKING RESULTS")
print("====================================")

print(f"R2   = {m_stack[0]:.4f}")
print(f"RMSE = {m_stack[1]:.4f}")
print(f"rRMSE= {m_stack[2]:.2f}%")
print(f"MAE  = {m_stack[3]:.4f}")

# =====================================================
# BASE MODEL METRICS
# =====================================================
results = []

for model_name in base_models:

    pred = base_test_predictions[model_name]

    m = calc_metric(y, pred)

    results.append([
        model_name,
        m[0],
        m[1],
        m[2],
        m[3]
    ])

results.append([
    "Stacking",
    m_stack[0],
    m_stack[1],
    m_stack[2],
    m_stack[3]
])

metric_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2",
        "RMSE",
        "rRMSE",
        "MAE"
    ]
)

# =====================================================
# SAVE METRICS
# =====================================================
metric_df.to_csv(
    os.path.join(metric_dir, "stacking_metrics.csv"),
    index=False
)

# =====================================================
# SAVE PREDICTIONS
# =====================================================
pred_df = pd.DataFrame({
    "Observed": y,
    "RF": base_test_predictions["RF"],
    "XGB": base_test_predictions["XGB"],
    "BPNN": base_test_predictions["BPNN"],
    "Stacking": stack_pred
})

pred_df.to_csv(
    os.path.join(metric_dir, "stacking_predictions.csv"),
    index=False
)

# =====================================================
# FIGURE 1
# =====================================================
plt.figure(figsize=(8, 5))

sns.barplot(
    data=metric_df,
    x="Model",
    y="R2"
)

plt.title("Model Comparison (R²)")
plt.ylabel("R²")

plt.tight_layout()

plt.savefig(
    os.path.join(fig_dir, "stacking_r2_comparison.png"),
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 2 PARITY
# =====================================================
plt.figure(figsize=(6, 6))

sns.scatterplot(
    x=y,
    y=stack_pred,
    s=60,
    alpha=0.7
)

x_line = np.linspace(
    min(y),
    max(y),
    100
)

plt.plot(
    x_line,
    x_line,
    "k--"
)

z = np.polyfit(y, stack_pred, 1)

p = np.poly1d(z)

plt.plot(
    x_line,
    p(x_line),
    "r-"
)

plt.xlabel("Observed Pn")
plt.ylabel("Predicted Pn")

plt.title(
    f"Stacking Parity Plot\nR²={m_stack[0]:.3f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(fig_dir, "stacking_parity.png"),
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 3 RESIDUAL
# =====================================================
residual = y - stack_pred

plt.figure(figsize=(8, 5))

sns.histplot(
    residual,
    kde=True,
    bins=30
)

plt.axvline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Residual")
plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(fig_dir, "stacking_residuals.png"),
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 4 OBSERVED VS ALL
# =====================================================
plt.figure(figsize=(10, 6))

x_axis = np.arange(len(y))

plt.plot(
    x_axis,
    y,
    linewidth=3,
    color="black",
    label="Observed"
)

for model_name in base_test_predictions:

    plt.plot(
        x_axis,
        base_test_predictions[model_name],
        alpha=0.6,
        linewidth=1.5,
        label=model_name
    )

plt.plot(
    x_axis,
    stack_pred,
    linewidth=3,
    color="red",
    label="Stacking"
)

plt.xlabel("Sample Index")
plt.ylabel("Pn")

plt.title("Observed vs Stacking Predictions")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(fig_dir, "stacking_all_predictions.png"),
    dpi=300
)

plt.close()

# =====================================================
# META FEATURE CORRELATION
# =====================================================
meta_df = pd.DataFrame(
    oof_predictions,
    columns=["RF", "XGB", "BPNN"]
)

corr = meta_df.corr()

plt.figure(figsize=(6, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".3f"
)

plt.title("Meta Feature Correlation")

plt.tight_layout()

plt.savefig(
    os.path.join(fig_dir, "stacking_meta_feature_corr.png"),
    dpi=300
)

plt.close()

# =====================================================
# SAVE FINAL MODELS
# =====================================================
print("\n====================================")
print("SAVING MODELS")
print("====================================")

# retrain full data
for model_name, model in base_models.items():

    model.fit(X, y)

    joblib.dump(
        model,
        os.path.join(
            model_dir,
            f"{model_name}_stacking.pkl"
        )
    )

joblib.dump(
    meta_model,
    os.path.join(
        model_dir,
        "meta_model_stacking.pkl"
    )
)

# =====================================================
# SAVE FEATURE NAMES
# =====================================================
np.save(
    os.path.join(model_dir, "stacking_feature_names.npy"),
    feature_names
)

# =====================================================
# DONE
# =====================================================
print("\n====================================")
print("STACKING COMPLETED")
print("====================================")

print("\nFigures:")
print(fig_dir)

print("\nMetrics:")
print(metric_dir)

print("\nModels:")
print(model_dir)