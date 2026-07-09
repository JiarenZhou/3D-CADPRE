import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== 参数 ==================
input_dir = r"I:\DJI_Workspace\V12-M350\final"  # LAS 文件夹
output_csv = os.path.join(input_dir, "Canopy_volume_V12.csv")
voxel_size = 0.01  # XY 栅格大小，单位 m，可根据点云稠密度调整

# ================== 冠层体积计算函数 ==================
def compute_canopy_volume(las_path, voxel_size=0.01):
    """
    计算单个 LAS 文件的冠层体积（m³）
    """
    las = laspy.read(las_path)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    # 局部坐标系：z 最小值设为 0
    z = z - z.min()

    # 网格化 XY
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_bins = np.arange(x_min, x_max + voxel_size, voxel_size)
    y_bins = np.arange(y_min, y_max + voxel_size, voxel_size)

    # 计算栅格中最大高度，累计体积
    volume = 0.0
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = (x >= x_bins[i]) & (x < x_bins[i+1]) & (y >= y_bins[j]) & (y < y_bins[j+1])
            if np.any(mask):
                h_max = z[mask].max()
                volume += h_max * voxel_size * voxel_size  # 栅格面积 * 高度

    return volume

# ================== 批量计算 ==================
las_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".las")]
results = []

print(f"📦 发现 {len(las_files)} 个 LAS 文件，开始计算冠层体积...")

for las_name in tqdm(las_files):
    las_path = os.path.join(input_dir, las_name)
    cv = compute_canopy_volume(las_path, voxel_size=voxel_size)
    results.append({
        "LAS_File": las_name,
        "Canopy_Volume_m3": cv
    })

# ================== 保存 CSV ==================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"🎉 冠层体积计算完成，结果保存至 {output_csv}")
