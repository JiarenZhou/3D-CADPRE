import warnings
warnings.filterwarnings("ignore")

import os
import copy
import random
import itertools

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from neural_models import BPNN, CNN1D

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
data_path = (
    r"C:\Users\Jeron\Desktop\code\反演\feature"
    r"\selected_features.xlsx"
)

metric_dir = (
    r"C:\Users\Jeron\Desktop\code\反演\outputs\metrics"
)

model_dir = (
    r"C:\Users\Jeron\Desktop\code\反演\outputs\models"
)

os.makedirs(metric_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(data_path)

y = df["Pn"].values.astype(np.float32)

X = df.drop(columns=["Pn"]).values.astype(np.float32)

# =====================================================
# CV
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# =====================================================
# METRIC
# =====================================================
def calc_metric(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    rrmse = rmse / np.mean(y_true) * 100

    mae = mean_absolute_error(y_true, y_pred)

    return r2, rmse, rrmse, mae

# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_one_model(
    model,
    X_train,
    y_train,
    X_valid,
    lr=0.001,
    epochs=200,
    patience=30
):

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

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_loss = np.inf
    best_state = None

    patience_counter = 0

    # =================================================
    # TRAIN
    # =================================================
    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        pred = model(X_train_tensor)

        loss = criterion(
            pred,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:

            best_loss = current_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:
            break

    # =================================================
    # LOAD BEST
    # =================================================
    model.load_state_dict(best_state)

    # =================================================
    # PREDICT
    # =================================================
    model.eval()

    with torch.no_grad():

        pred_valid = model(
            X_valid_tensor
        ).numpy().flatten()

    return pred_valid

# =====================================================
# PARAM GRID
# =====================================================
bpnn_grid = list(itertools.product(

    [8, 16, 32, 64, 128],

    [1e-4, 1e-3, 1e-2],

    [20, 30]
))

cnn_grid = list(itertools.product(

    [8, 16, 32],

    [16, 32, 64],

    [16, 32, 64],

    [1e-4, 1e-3, 1e-2],

    [20, 30]
))

total_steps = len(bpnn_grid) + len(cnn_grid)

results = []

# =====================================================
# BEST MODEL TRACKING
# =====================================================
best_bpnn_r2 = -np.inf
best_bpnn_model = None
best_bpnn_info = None

best_cnn_r2 = -np.inf
best_cnn_model = None
best_cnn_info = None

# =====================================================
# GRID SEARCH
# =====================================================
with tqdm(total=total_steps, desc="DL Grid Search") as pbar:

    # =================================================
    # BPNN
    # =================================================
    for hidden, lr, patience in bpnn_grid:

        oof_pred = np.zeros(len(y))

        for train_idx, valid_idx in kf.split(X):

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_valid = X[valid_idx]

            model = BPNN(
                X.shape[1],
                hidden
            )

            pred_valid = train_one_model(
                model,
                X_train,
                y_train,
                X_valid,
                lr=lr,
                patience=patience
            )

            oof_pred[valid_idx] = pred_valid

        # =============================================
        # METRICS
        # =============================================
        r2, rmse, rrmse, mae = calc_metric(
            y,
            oof_pred
        )

        results.append({

            "Model": "BPNN",

            "Hidden": hidden,

            "Conv1": None,
            "Conv2": None,
            "Dense": None,

            "LR": lr,
            "Patience": patience,

            "R2": r2,
            "RMSE": rmse,
            "rRMSE(%)": rrmse,
            "MAE": mae
        })

        # =============================================
        # BEST
        # =============================================
        if r2 > best_bpnn_r2:

            best_bpnn_r2 = r2

            best_bpnn_model = model

            best_bpnn_info = {
                "Hidden": hidden,
                "LR": lr,
                "Patience": patience
            }

        pbar.update(1)

    # =================================================
    # CNN
    # =================================================
    for conv1, conv2, dense, lr, patience in cnn_grid:

        oof_pred = np.zeros(len(y))

        for train_idx, valid_idx in kf.split(X):

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_valid = X[valid_idx]

            model = CNN1D(
                conv1,
                conv2,
                dense,
                2
            )

            pred_valid = train_one_model(
                model,
                X_train,
                y_train,
                X_valid,
                lr=lr,
                patience=patience
            )

            oof_pred[valid_idx] = pred_valid

        # =============================================
        # METRICS
        # =============================================
        r2, rmse, rrmse, mae = calc_metric(
            y,
            oof_pred
        )

        results.append({

            "Model": "CNN",

            "Hidden": None,

            "Conv1": conv1,
            "Conv2": conv2,
            "Dense": dense,

            "LR": lr,
            "Patience": patience,

            "R2": r2,
            "RMSE": rmse,
            "rRMSE(%)": rrmse,
            "MAE": mae
        })

        # =============================================
        # BEST
        # =============================================
        if r2 > best_cnn_r2:

            best_cnn_r2 = r2

            best_cnn_model = model

            best_cnn_info = {
                "Conv1": conv1,
                "Conv2": conv2,
                "Dense": dense,
                "LR": lr,
                "Patience": patience
            }

        pbar.update(1)

# =====================================================
# SAVE CSV
# =====================================================
out = pd.DataFrame(results)

out = out.sort_values(
    by="R2",
    ascending=False
)

csv_path = (
    r"C:\Users\Jeron\Desktop\code\反演\outputs\metrics"
    r"\dl_selected_results.csv"
)

out.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# SAVE BEST MODELS
# =====================================================
bpnn_save_path = (
    r"C:\Users\Jeron\Desktop\code\反演\outputs\models"
    r"\best_bpnn_oof.pth"
)

cnn_save_path = (
    r"C:\Users\Jeron\Desktop\code\反演\outputs\models"
    r"\best_cnn_oof.pth"
)

if best_bpnn_model is not None:

    torch.save(
        best_bpnn_model.state_dict(),
        bpnn_save_path
    )

if best_cnn_model is not None:

    torch.save(
        best_cnn_model.state_dict(),
        cnn_save_path
    )

# =====================================================
# PRINT
# =====================================================
print("\n==============================")
print("TOP MODELS")
print("==============================")

print(out.head(10))

print("\n==============================")
print("BEST BPNN")
print("==============================")

print(best_bpnn_info)
print(f"R2 = {best_bpnn_r2:.6f}")

print("\n==============================")
print("BEST CNN")
print("==============================")

print(best_cnn_info)
print(f"R2 = {best_cnn_r2:.6f}")

print("\nSaved metrics:")
print(csv_path)

print("\nSaved models:")
print(bpnn_save_path)
print(cnn_save_path)