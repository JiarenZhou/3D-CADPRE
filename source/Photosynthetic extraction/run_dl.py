import itertools
import pandas as pd
from tqdm import tqdm

from preprocess import load_data
from neural_models import BPNN, CNN1D
from train_nn import train_model

# =====================================================
# load data
# =====================================================
X, y, _ = load_data()

results = []

# =====================================================
# grids
# =====================================================
bpnn_grid = list(itertools.product(
    [8, 16, 32, 64, 128],
    [1e-4, 1e-3, 1e-2]
))

cnn_grid = list(itertools.product(
    [8, 16, 32],        # conv1
    [16, 32, 64],       # conv2
    [16, 32, 64],       # dense
    [1e-4, 1e-3, 1e-2]  # lr
))

total_steps = len(bpnn_grid) + len(cnn_grid)

# =====================================================
# training loop
# =====================================================
with tqdm(total=total_steps, desc="Grid Search Progress") as pbar:

    # =====================================================
    # BPNN
    # =====================================================
    for h, lr in bpnn_grid:

        model = BPNN(
            X.shape[1],
            h
        )

        m = train_model(
            model,
            X,
            y,
            lr
        )

        results.append([
            "BPNN",
            h,
            lr,
            m[0],
            m[1],
            m[2],
            m[3]
        ])

        pbar.update(1)

    # =====================================================
    # CNN
    # =====================================================
    for conv1, conv2, dense, lr in cnn_grid:

        # ⚠️ 关键修复：CNN1D 只接受位置参数
        model = CNN1D(
            conv1,
            conv2,
            dense,
            2   # kernel_size
        )

        m = train_model(
            model,
            X,
            y,
            lr,
            epochs=80
        )

        results.append([
            "CNN",
            conv1,
            conv2,
            dense,
            lr,
            m[0],
            m[1],
            m[2],
            m[3]
        ])

        pbar.update(1)

# =====================================================
# save results
# =====================================================
out = pd.DataFrame(results)

out.to_csv(
    r"outputs/metrics/dl_results.csv",
    index=False
)

print(out)