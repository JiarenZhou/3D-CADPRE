import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

# =====================================================
# GET MODELS (NO PCA VERSION)
# =====================================================
def get_models():

    models = {}

    # =================================================
    # MLR
    # =================================================
    mlr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    mlr_param = {
        "model__fit_intercept": [True, False]
    }

    models["MLR"] = (mlr_pipe, mlr_param)

    # =================================================
    # RF
    # =================================================
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            random_state=42
        ))
    ])

    rf_param = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    models["RF"] = (rf_pipe, rf_param)

    # =================================================
    # SVR
    # =================================================
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])

    svr_param = {
        "model__kernel": ["rbf"],
        "model__C": [1, 10, 100],
        "model__gamma": ["scale", "auto"],
        "model__epsilon": [0.1, 0.5]
    }

    models["SVR"] = (svr_pipe, svr_param)

    # =================================================
    # XGB
    # =================================================
    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        ))
    ])

    xgb_param = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3]
    }

    models["XGB"] = (xgb_pipe, xgb_param)

    return models
