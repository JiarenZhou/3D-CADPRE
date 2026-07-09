import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import tempfile

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict
)

from metrics import calc_metric
from traditional_models import get_models

# =====================================================
# load dataset
# =====================================================
df = pd.read_excel(
    "C:/Users/Jeron/Desktop/code/反演/feature/selected_features.xlsx"
)

y = df["Pn"].values
X = df.drop(columns=["Pn"]).values

# =====================================================
# cross validation
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = get_models()

results = []

# =====================================================
# training
# =====================================================
for name, (pipe, param) in models.items():

    print(f"\n========== {name} ==========")

    grid = GridSearchCV(
        pipe,
        param,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # =====================================================
    # training time
    # =====================================================
    start_train = time.time()

    grid.fit(X, y)

    train_time = time.time() - start_train

    # =====================================================
    # prediction time
    # =====================================================
    start_pred = time.time()

    pred = cross_val_predict(
        grid.best_estimator_,
        X,
        y,
        cv=kf,
        n_jobs=-1
    )

    pred_time = time.time() - start_pred

    # =====================================================
    # metrics
    # =====================================================
    m = calc_metric(y, pred)

    # =====================================================
    # model size
    # =====================================================
    with tempfile.NamedTemporaryFile(delete=False) as tmp:

        joblib.dump(
            grid.best_estimator_,
            tmp.name
        )

        model_size_mb = (
            os.path.getsize(tmp.name)
            / 1024 / 1024
        )

    # =====================================================
    # PCA dimension
    # =====================================================
    try:
        pca_dim = (
            grid.best_estimator_
            .named_steps["pca"]
            .n_components_
        )
    except:
        pca_dim = None

    # =====================================================
    # save results
    # =====================================================
    results.append([

        name,

        round(m[0], 4),   # R2
        round(m[1], 4),   # RMSE
        round(m[2], 2),   # rRMSE
        round(m[3], 4),   # MAE

        round(train_time, 3),
        round(pred_time, 3),

        round(model_size_mb, 3),

        pca_dim,

        str(grid.best_params_)
    ])

    # =====================================================
    # print
    # =====================================================
    print("Best Params:")
    print(grid.best_params_)

    print(
        f"R2={m[0]:.4f} "
        f"RMSE={m[1]:.4f} "
        f"rRMSE={m[2]:.2f}% "
        f"MAE={m[3]:.4f}"
    )

    print(
        f"Train Time: {train_time:.3f}s"
    )

    print(
        f"Predict Time: {pred_time:.3f}s"
    )

    print(
        f"Model Size: {model_size_mb:.3f} MB"
    )

    print(
        f"PCA Dim: {pca_dim}"
    )

# =====================================================
# save csv
# =====================================================
out = pd.DataFrame(
    results,
    columns=[

        "Model",

        "R2",
        "RMSE",
        "rRMSE(%)",
        "MAE",

        "Train_Time(s)",
        "Predict_Time(s)",

        "Model_Size(MB)",

        "PCA_Dim",

        "Best_Params"
    ]
)

out.to_csv(
    r"C:/Users/Jeron/Desktop/code/反演/outputs/metrics/ml_selected_results.csv",
    index=False
)

print("\nSaved")