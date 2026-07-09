import os
import rasterio
import numpy as np

# 四个波段的文件夹
bands = ["Green", "Red", "RedEdge", "NIR"]
base_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots"
out_base_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots_final"

# NDVI阈值
ndvi_threshold = 0.7

# 创建输出文件夹
for band in bands:
    os.makedirs(os.path.join(out_base_dir, band), exist_ok=True)

# 获取所有小区名称（以Red波段为参考）
red_dir = os.path.join(base_dir, "Red")
file_list = sorted([f for f in os.listdir(red_dir) if f.lower().endswith(".tif")])

for fname in file_list:
    try:
        # 读取Red和NIR波段
        with rasterio.open(os.path.join(base_dir, "Red", fname)) as src_red:
            red = src_red.read(1).astype(np.float32)
            profile = src_red.profile

        with rasterio.open(os.path.join(base_dir, "NIR", fname)) as src_nir:
            nir = src_nir.read(1).astype(np.float32)

        # 计算NDVI
        ndvi = (nir - red) / (nir + red + 1e-6)  # 避免除零

        # 土壤掩膜
        mask = ndvi >= ndvi_threshold  # True表示植被，False表示土壤

        # 对四个波段进行掩膜
        for band in bands:
            band_path = os.path.join(base_dir, band, fname)
            out_path = os.path.join(out_base_dir, band, fname)

            with rasterio.open(band_path) as src:
                data = src.read(1).astype(np.float32)

            # 掩膜处理
            data_masked = np.where(mask, data, 0)

            # 保存
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data_masked.astype(profile['dtype']), 1)

        print(f"Processed: {fname}")

    except Exception as e:
        print(f"Error processing {fname}: {e}")