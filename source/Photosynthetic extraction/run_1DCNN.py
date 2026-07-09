import warnings
warnings.filterwarnings("ignore")

import os
import time
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    KFold
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.preprocessing import (
    StandardScaler
)

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# =====================================================
# GLOBAL STYLE
# =====================================================
FONT_SIZE = 28

plt.rcParams.update({

    "font.family": "Arial",
    "axes.unicode_minus": False,

    # 全局字体
    "font.size": FONT_SIZE,

    # 坐标轴
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,

    # 刻度
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,

    # 图例
    "legend.fontsize": FONT_SIZE,

    # Figure标题
    "figure.titlesize": FONT_SIZE,

    # 保存
    "figure.dpi": 330,
    "savefig.dpi": 330

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
# DEVICE
# =====================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

# =====================================================
# PATH
# =====================================================
data_path = (
    r"C:/Users/Jeron/Desktop/code/反演/feature/"
    r"selected_features.xlsx"
)

metric_dir = (
    r"C:/Users/Jeron/Desktop/code/反演/outputs/metrics"
)

fig_dir = (
    r"C:/Users/Jeron/Desktop/code/反演/outputs/figures"
)

os.makedirs(metric_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(data_path)

y = df["Pn"].values.astype(np.float32)

X = df.drop(columns=["Pn"]).values.astype(np.float32)

feature_names = (
    df.drop(columns=["Pn"]).columns.tolist()
)

print("\n====================================")
print("DATA INFO")
print("====================================")

print(f"Samples  : {X.shape[0]}")
print(f"Features : {X.shape[1]}")

# =====================================================
# STANDARDIZATION
# CNN 对输入尺度非常敏感
# RF/XGB不敏感，但CNN必须标准化
# =====================================================
scaler = StandardScaler()

X = scaler.fit_transform(X)

# =====================================================
# METRIC FUNCTION
# =====================================================
def calc_metric(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    rrmse = (
        rmse / np.mean(y_true) * 100
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return r2, rmse, rrmse, mae

# =====================================================
# IMPROVED 1D CNN
# =====================================================
class ImprovedCNN1D(nn.Module):

    def __init__(
        self,
        input_length,
        conv1=32,
        conv2=64,
        conv3=128,
        dense1=128,
        dense2=64,
        dropout=0.3
    ):

        super().__init__()

        # =============================================
        # FEATURE EXTRACTOR
        # =============================================
        self.features = nn.Sequential(

            # -----------------------------------------
            # BLOCK 1
            # -----------------------------------------
            nn.Conv1d(
                in_channels=1,
                out_channels=conv1,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(conv1),

            nn.ReLU(),

            nn.MaxPool1d(2),

            # -----------------------------------------
            # BLOCK 2
            # -----------------------------------------
            nn.Conv1d(
                conv1,
                conv2,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(conv2),

            nn.ReLU(),

            nn.MaxPool1d(2),

            # -----------------------------------------
            # BLOCK 3
            # -----------------------------------------
            nn.Conv1d(
                conv2,
                conv3,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(conv3),

            nn.ReLU(),

            # global pooling
            nn.AdaptiveAvgPool1d(1)
        )

        # =============================================
        # REGRESSOR
        # =============================================
        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(conv3, dense1),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(dense1, dense2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(dense2, 1)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.features(x)

        x = self.regressor(x)

        return x

# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_fold(
    model,
    train_loader,
    valid_loader,
    epochs=300,
    lr=0.001,
    patience=40
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    best_loss = np.inf

    patience_counter = 0

    best_state = None

    train_losses = []
    valid_losses = []

    # =================================================
    # TRAIN LOOP
    # =================================================
    for epoch in range(epochs):

        # =============================================
        # TRAIN
        # =============================================
        model.train()

        epoch_train_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5
            )

            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # =============================================
        # VALID
        # =============================================
        model.eval()

        epoch_valid_loss = 0

        with torch.no_grad():

            for xb, yb in valid_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                loss = criterion(pred, yb)

                epoch_valid_loss += loss.item()

        epoch_valid_loss /= len(valid_loader)

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

        scheduler.step(epoch_valid_loss)

        # =============================================
        # EARLY STOPPING
        # =============================================
        if epoch_valid_loss < best_loss:

            best_loss = epoch_valid_loss

            best_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            break

    # =================================================
    # LOAD BEST MODEL
    # =================================================
    model.load_state_dict(best_state)

    return model, train_losses, valid_losses

# =====================================================
# CROSS VALIDATION
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# =====================================================
# PARAMETER SET
# =====================================================
param_grid = [

    {
        "conv1": 32,
        "conv2": 64,
        "conv3": 128,
        "dense1": 128,
        "dense2": 64,
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 16
    },

    {
        "conv1": 32,
        "conv2": 64,
        "conv3": 128,
        "dense1": 256,
        "dense2": 128,
        "dropout": 0.3,
        "lr": 0.001,
        "batch_size": 16
    },

    {
        "conv1": 64,
        "conv2": 128,
        "conv3": 256,
        "dense1": 256,
        "dense2": 128,
        "dropout": 0.4,
        "lr": 0.0005,
        "batch_size": 16
    }
]

# =====================================================
# SEARCH
# =====================================================
results = []

best_r2 = -np.inf

best_pred = None

best_params = None

best_loss_curve = None

# =====================================================
# SEARCH LOOP
# =====================================================
for idx, params in enumerate(param_grid):

    print("\n====================================")
    print(f"MODEL {idx + 1}")
    print("====================================")

    print(params)

    oof_pred = np.zeros(len(y))

    fold_train_losses = []
    fold_valid_losses = []

    start_time = time.time()

    # =================================================
    # CV LOOP
    # =================================================
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(X)
    ):

        print(f"\nFold {fold + 1}")

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        # =============================================
        # TENSOR
        # =============================================
        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y_train_tensor = torch.tensor(
            y_train.reshape(-1, 1),
            dtype=torch.float32
        )

        X_valid_tensor = torch.tensor(
            X_valid,
            dtype=torch.float32
        )

        y_valid_tensor = torch.tensor(
            y_valid.reshape(-1, 1),
            dtype=torch.float32
        )

        # =============================================
        # LOADER
        # =============================================
        train_loader = DataLoader(
            TensorDataset(
                X_train_tensor,
                y_train_tensor
            ),
            batch_size=params["batch_size"],
            shuffle=True
        )

        valid_loader = DataLoader(
            TensorDataset(
                X_valid_tensor,
                y_valid_tensor
            ),
            batch_size=params["batch_size"],
            shuffle=False
        )

        # =============================================
        # MODEL
        # =============================================
        model = ImprovedCNN1D(
            input_length=X.shape[1],
            conv1=params["conv1"],
            conv2=params["conv2"],
            conv3=params["conv3"],
            dense1=params["dense1"],
            dense2=params["dense2"],
            dropout=params["dropout"]
        ).to(device)

        # =============================================
        # TRAIN
        # =============================================
        model, train_losses, valid_losses = train_fold(
            model,
            train_loader,
            valid_loader,
            epochs=300,
            lr=params["lr"],
            patience=40
        )

        fold_train_losses.append(train_losses)
        fold_valid_losses.append(valid_losses)

        # =============================================
        # PREDICT
        # =============================================
        model.eval()

        with torch.no_grad():

            pred = model(
                X_valid_tensor.to(device)
            )

        pred = pred.cpu().numpy().flatten()

        oof_pred[valid_idx] = pred

    # =================================================
    # METRICS
    # =================================================
    r2, rmse, rrmse, mae = calc_metric(
        y,
        oof_pred
    )

    elapsed = time.time() - start_time

    results.append([

        idx + 1,

        r2,
        rmse,
        rrmse,
        mae,

        elapsed,

        str(params)
    ])

    print("\n====================================")
    print("RESULT")
    print("====================================")

    print(f"R2    = {r2:.4f}")
    print(f"RMSE  = {rmse:.4f}")
    print(f"rRMSE = {rrmse:.2f}%")
    print(f"MAE   = {mae:.4f}")

    # =================================================
    # BEST
    # =================================================
    if r2 > best_r2:

        best_r2 = r2

        best_pred = oof_pred.copy()

        best_params = params

        best_loss_curve = (
            fold_train_losses,
            fold_valid_losses
        )

# =====================================================
# SAVE RESULT TABLE
# =====================================================
result_df = pd.DataFrame(
    results,
    columns=[

        "Model_ID",

        "R2",
        "RMSE",
        "rRMSE(%)",
        "MAE",

        "Time(s)",

        "Params"
    ]
)

result_df = result_df.sort_values(
    "R2",
    ascending=False
)

csv_path = os.path.join(
    metric_dir,
    "improved_cnn_results.csv"
)

result_df.to_csv(
    csv_path,
    index=False
)

# =====================================================
# BEST METRIC
# =====================================================
best_metric = calc_metric(y, best_pred)

print("\n====================================")
print("BEST MODEL")
print("====================================")

print(best_params)

print(f"\nBEST R2 = {best_metric[0]:.4f}")

# =====================================================
# FIGURE 1
# PARITY
# =====================================================
plt.figure(figsize=(6, 5))

sns.scatterplot(
    x=y,
    y=best_pred,
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
    "k--",
    linewidth=1.5
)

z = np.polyfit(y, best_pred, 1)

p = np.poly1d(z)

plt.plot(
    x_line,
    p(x_line),
    "r-",
    linewidth=2
)

plt.xlabel("Observed Pn")
plt.ylabel("Predicted Pn")

plt.title(
    f"Improved CNN1D\n"
    f"R²={best_metric[0]:.3f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        fig_dir,
        "improved_cnn_parity.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 2
# RESIDUAL
# =====================================================
residual = y - best_pred

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
    os.path.join(
        fig_dir,
        "improved_cnn_residual.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 3
# LOSS CURVE
# =====================================================
train_losses = best_loss_curve[0][0]
valid_losses = best_loss_curve[1][0]

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    valid_losses,
    label="Valid Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Training Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        fig_dir,
        "improved_cnn_loss_curve.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# DONE
# =====================================================
print("\n====================================")
print("FINISHED")
print("====================================")

print(f"\nCSV:\n{csv_path}")

print(f"\nFigures:\n{fig_dir}")