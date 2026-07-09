import numpy as np

RANDOM_STATE = 42

DATA_PATH = r"../data/光谱+辐射+纹理+表型反演光合参数数据集.xlsx"

TARGET = "Pn"

KFOLD = 5

DL_KFOLD = 3

PCA_LIST = [0.85,0.90,0.95]

FEATURE_COLS = None

METRIC_COLUMNS = [
    "Model",
    "R2",
    "RMSE",
    "rRMSE(%)",
    "MAE"
]

np.random.seed(RANDOM_STATE)