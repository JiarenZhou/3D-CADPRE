import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为 Arial
rcParams['font.family'] = 'Arial'

bands = ["Green", "Red", "RedEdge", "NIR"]
base_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots_final"

# 输出文件夹
single_band_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots_visual"
panel_dir = os.path.join(single_band_dir, "total")

os.makedirs(single_band_dir, exist_ok=True)
os.makedirs(panel_dir, exist_ok=True)

# 以Red波段文件名为参考
red_dir = os.path.join(base_dir, "Red")
file_list = sorted([f for f in os.listdir(red_dir) if f.lower().endswith(".tif")])

# -----------------------------
# Step 1: 计算全局最大绝对值
# -----------------------------
global_max_abs = 0.0
for fname in file_list:
    for band in bands:
        band_path = os.path.join(base_dir, band, fname)
        with rasterio.open(band_path) as src:
            data = src.read(1).astype(np.float32)
        band_max_abs = np.max(np.abs(data))
        if band_max_abs > global_max_abs:
            global_max_abs = band_max_abs

print(f"Global maximum absolute value: {global_max_abs}")

# -----------------------------
# Step 2: 批量生成可视化
# -----------------------------
for fname in file_list:
    try:
        # 读取所有波段
        band_data = {}
        for band in bands:
            band_path = os.path.join(base_dir, band, fname)
            with rasterio.open(band_path) as src:
                data = src.read(1).astype(np.float32)
            # 全局归一化
            if global_max_abs > 0:
                norm_data = data / global_max_abs
            else:
                norm_data = data
            band_data[band] = norm_data

            # --- 单波段可视化 ---
            plt.figure(figsize=(8,6))
            im = plt.imshow(norm_data, cmap='seismic', vmin=-1, vmax=1)
            plt.axis('off')
            plt.title(f"{band} - {fname}", fontsize=12)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Reflectance (-1 to 1)", fontsize=10)

            single_out = os.path.join(single_band_dir, f"{os.path.splitext(fname)[0]}_{band}.png")
            plt.savefig(single_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved single band: {single_out}")

        # --- 1x4 面板可视化 ---
        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        plt.subplots_adjust(wspace=0.05)
        for i, band in enumerate(bands):
            axes[i].imshow(band_data[band], cmap='seismic', vmin=-1, vmax=1)
            axes[i].set_title(band, fontsize=12)
            axes[i].axis('off')

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-1, vmax=1)),
            ax=axes.ravel().tolist(),
            fraction=0.03, pad=0.02
        )
        cbar.set_label("Reflectance (-1 to 1)", fontsize=12)

        panel_out = os.path.join(panel_dir, f"{os.path.splitext(fname)[0]}.png")
        plt.savefig(panel_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved panel visualization: {panel_out}")

    except Exception as e:
        print(f"Error processing {fname}: {e}")