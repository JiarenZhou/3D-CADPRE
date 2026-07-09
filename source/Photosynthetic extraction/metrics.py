import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

def calc_metric(y,p):

    rmse = np.sqrt(
        mean_squared_error(y,p)
    )

    rrmse = (rmse / np.mean(np.abs(y))) * 100

    mae = mean_absolute_error(y,p)

    r2 = r2_score(y,p)

    return [
        r2,
        rmse,
        rrmse,
        mae
    ]