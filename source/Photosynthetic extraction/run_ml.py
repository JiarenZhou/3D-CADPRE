import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict
)

from metrics import calc_metric
from preprocess import load_data
from traditional_models import get_models

X,y,_ = load_data()

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = get_models()

results = []

for name,(pipe,param) in models.items():

    print(f"\n========== {name} ==========")

    grid = GridSearchCV(
        pipe,
        param,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X,y)

    pred = cross_val_predict(
        grid.best_estimator_,
        X,
        y,
        cv=kf,
        n_jobs=-1
    )

    m = calc_metric(y,pred)

    results.append([
        name,
        round(m[0],4),
        round(m[1],4),
        round(m[2],2),
        round(m[3],4)
    ])

    print(grid.best_params_)

    print(
        f"R2={m[0]:.4f} "
        f"RMSE={m[1]:.4f} "
        f"rRMSE={m[2]:.2f}% "
        f"MAE={m[3]:.4f}"
    )

out = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2",
        "RMSE",
        "rRMSE(%)",
        "MAE"
    ]
)

out.to_csv(
    r"C:/Users/Jeron/Desktop/code/反演/outputs/metrics/ml_results.csv",
    index=False
)

print("\nSaved")