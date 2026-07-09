import os
import cv2
import rasterio
import numpy as np
import pandas as pd

from tqdm import tqdm

from skimage.feature import graycomatrix,graycoprops
from skimage.measure import shannon_entropy

# =========================================================
# path
# =========================================================

root = r"G:\2025公主岭无人机数据\M3M\M3M-V12"

folders = {
    "Green":"Green",
    "Red":"Red",
    "RedEdge":"RedEdge",
    "NIR":"NIR"
}

save_path = r"C:\Users\Jeron\Desktop\texture_features.csv"

# =========================================================
# glcm
# =========================================================

def texture_feature(img):

    img = img.astype(np.float32)

    img[np.isnan(img)] = 0

    # normalize -> uint8

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    feat = {}

    feat["contrast"] = graycoprops(glcm,"contrast")[0,0]

    feat["dissimilarity"] = graycoprops(glcm,"dissimilarity")[0,0]

    feat["homogeneity"] = graycoprops(glcm,"homogeneity")[0,0]

    feat["energy"] = graycoprops(glcm,"energy")[0,0]

    feat["correlation"] = graycoprops(glcm,"correlation")[0,0]

    feat["ASM"] = graycoprops(glcm,"ASM")[0,0]

    feat["entropy"] = shannon_entropy(img)

    return feat

# =========================================================
# file list
# =========================================================

green_dir = os.path.join(
    root,
    folders["Green"]
)

file_list = sorted([
    f for f in os.listdir(green_dir)
    if f.lower().endswith((".tif",".tiff"))
])

# =========================================================
# extract
# =========================================================

results = []

for file in tqdm(file_list):

    row = {
        "file_name":file
    }

    for band,folder in folders.items():

        path = os.path.join(
            root,
            folder,
            file
        )

        if not os.path.exists(path):

            print(f"missing: {path}")

            continue

        try:

            with rasterio.open(path) as src:

                img = src.read(1).astype(np.float32)

        except Exception as e:

            print(f"error: {path}")
            print(e)

            continue

        img[np.isnan(img)] = 0

        img = cv2.resize(
            img,
            (128,128),
            interpolation=cv2.INTER_CUBIC
        )

        feat = texture_feature(img)

        for k,v in feat.items():

            row[f"{band}_{k}"] = v

    results.append(row)

# =========================================================
# save
# =========================================================

out = pd.DataFrame(results)

out.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n========================")
print("Texture extraction done")
print("========================")

print("\nSaved:")
print(save_path)