import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_raster

# =========================
# 路径
# =========================
input_path = r"H:\2026公主岭无人机数据\M3T\V6-M3T\plots_final"
output_path = r"H:\2026公主岭无人机数据\M3T\V6-M3T\plots_visual"

os.makedirs(output_path, exist_ok=True)

# =========================
# 固定温度范围
# =========================
vmin = 25.0
vmax = 35.0

print(f"Using fixed range: {vmin} - {vmax} °C")

# =========================
# colormap（绿→黄→红）
# =========================
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "gyr", ["green", "yellow", "red"]
)

# =========================
# 处理
# =========================
for file in os.listdir(input_path):
    if file.lower().endswith(".tif"):

        in_file = os.path.join(input_path, file)
        out_file = os.path.join(output_path, file)

        with rasterio.open(in_file) as src:
            img = src.read(1).astype(np.float32)
            profile = src.profile

        # =========================
        # 记录 NaN mask
        # =========================
        nan_mask = np.isnan(img)

        # =========================
        # 归一化（固定范围）
        # =========================
        norm = (img - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)

        # =========================
        # colormap 映射
        # =========================
        rgba = cmap(norm)  # (H, W, 4)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        # =========================
        # 👉 NaN 设为白色
        # =========================
        rgb[nan_mask] = [255, 255, 255]

        # 转为 rasterio 格式
        rgb_raster = reshape_as_raster(rgb)

        # =========================
        # 更新 GeoTIFF 信息
        # =========================
        profile.update({
            "count": 3,
            "dtype": "uint8"
        })

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(rgb_raster)

        # =========================
        # PNG（带 colorbar）
        # =========================
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

        # 👉 设置 NaN 显示为白色（matplotlib）
        cmap.set_bad(color='white')

        ax.set_title(file)
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature (°C)")

        png_file = out_file.replace(".tif", "_vis.png")
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(file, "done")

print("All finished.")