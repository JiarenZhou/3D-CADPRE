import os
import rasterio
import numpy as np
import csv

# 四个波段文件夹
bands = ["Green", "Red", "RedEdge", "NIR"]
base_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots_final"
csv_path = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\reflectivity_means.csv"

# 以Red波段文件名为参考
red_dir = os.path.join(base_dir, "Red")
file_list = sorted([f for f in os.listdir(red_dir) if f.lower().endswith(".tif")])

# -----------------------------
# 自动检测 scale
# -----------------------------
def detect_scale(sample_file):
    with rasterio.open(sample_file) as src:
        data = src.read(1).astype(np.float32)

    max_val = np.nanmax(data)

    if max_val > 1000:
        scale = 10000.0
    elif max_val > 10:
        scale = 100.0
    else:
        scale = 1.0

    print("Scale detection:")
    print("Sample max value:", max_val)
    print("Detected scale factor:", scale)
    print()

    return scale


# 使用第一幅图检测
sample_path = os.path.join(base_dir, "Red", file_list[0])
scale_factor = detect_scale(sample_path)

# -----------------------------
# 开始统计
# -----------------------------
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    header = ["Plot"]
    for band in bands:
        header.append(f"{band}_mean")
    for band in bands:
        header.append(f"{band}_std")

    VI_names = ["NDVI", "GNDVI", "LCI", "NDRE", "OSAVI"]
    for vi in VI_names:
        header.append(f"{vi}_mean")
    for vi in VI_names:
        header.append(f"{vi}_std")

    writer.writerow(header)

    for fname in file_list:
        band_data = {}

        # 读取并归一化
        for band in bands:
            band_path = os.path.join(base_dir, band, fname)

            with rasterio.open(band_path) as src:
                data = src.read(1).astype(np.float32)

            # 自动scale
            data = data / scale_factor
            band_data[band] = data

        # -------------------
        # 波段统计
        # -------------------
        band_means = []
        band_stds = []

        for band in bands:
            data = band_data[band]
            mask = data > 0

            if np.any(mask):
                band_means.append(data[mask].mean())
                band_stds.append(data[mask].std())
            else:
                band_means.append(0.0)
                band_stds.append(0.0)

        # -------------------
        # 植被指数
        # -------------------
        Red = band_data["Red"]
        NIR = band_data["NIR"]
        Green = band_data["Green"]
        RedEdge = band_data["RedEdge"]

        eps = 1e-6

        NDVI = np.where((NIR + Red) > eps, (NIR - Red) / (NIR + Red + eps), np.nan)
        GNDVI = np.where((NIR + Green) > eps, (NIR - Green) / (NIR + Green + eps), np.nan)
        LCI = np.where((RedEdge + Red) > eps, (RedEdge - Red) / (RedEdge + Red + eps), np.nan)
        NDRE = np.where((NIR + RedEdge) > eps, (NIR - RedEdge) / (NIR + RedEdge + eps), np.nan)
        OSAVI = np.where((NIR + Red + 0.16) > eps, (NIR - Red) / (NIR + Red + 0.16), np.nan)

        VI_arrays = [NDVI, GNDVI, LCI, NDRE, OSAVI]

        VI_means = [np.nanmean(v) for v in VI_arrays]
        VI_stds = [np.nanstd(v) for v in VI_arrays]

        plot_name = os.path.splitext(fname)[0]

        writer.writerow(
            [plot_name]
            + band_means
            + band_stds
            + VI_means
            + VI_stds
        )

        print(f"Processed {plot_name}")

print("结果已保存到:", csv_path)