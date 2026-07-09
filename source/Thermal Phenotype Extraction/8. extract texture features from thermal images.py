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

root = r"G:\2025公主岭无人机数据\M3T\M3T-V12"

save_path = r"C:\Users\Jeron\Desktop\thermal_texture_features.csv"

# =========================================================
# glcm texture
# =========================================================

def texture_feature(img):

    img = img.astype(np.float32)

    img[np.isnan(img)] = 0

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
# tif list
# =========================================================

file_list = sorted([
    f for f in os.listdir(root)
    if f.lower().endswith((".tif",".tiff"))
])

# =========================================================
# extract
# =========================================================

results = []

for file in tqdm(file_list):

    path = os.path.join(root,file)

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

    row = {
        "file_name":file
    }

    for k,v in feat.items():

        row[f"Thermal_{k}"] = v

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
print("Thermal texture extraction done")
print("========================")

print("\nSaved:")
print(save_path)