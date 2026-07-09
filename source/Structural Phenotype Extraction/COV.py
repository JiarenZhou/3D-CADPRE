import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_cov_las(las_path, voxel_size=0.001):
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T

    # 过滤无效点
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]

    if len(xyz) == 0:
        return 0, 0.0

    # 最小外接长方体
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)

    # 占用 voxel 坐标
    voxel_coords = np.floor((xyz - min_xyz) / voxel_size).astype(int)
    voxel_coords_unique = np.unique(voxel_coords, axis=0)
    cov_count = voxel_coords_unique.shape[0]

    # 总 voxel 数量
    nx = int(np.ceil((max_xyz[0] - min_xyz[0]) / voxel_size))
    ny = int(np.ceil((max_xyz[1] - min_xyz[1]) / voxel_size))
    nz = int(np.ceil((max_xyz[2] - min_xyz[2]) / voxel_size))
    total_voxels = nx * ny * nz

    occupancy_ratio = cov_count / total_voxels if total_voxels > 0 else 0.0

    return cov_count, occupancy_ratio

def batch_calculate_cov_las(folder_path, output_csv_path, voxel_size=0.001):
    las_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.las')]
    results = []

    for las_file in tqdm(las_files, desc="计算 COV"):
        las_path = os.path.join(folder_path, las_file)
        cov_count, occupancy_ratio = calculate_cov_las(las_path, voxel_size)
        results.append({
            'LAS_File': las_file,
            'COV_Count': cov_count,
            'Occupancy_Ratio': occupancy_ratio
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"COV 计算完成，结果保存至 {output_csv_path}")


# ================== 使用示例 ==================
folder_path = r'G:\2025公主岭无人机数据\M350\V12-M350\final'
output_csv_path = r'C:\Users\Jeron\Desktop\COV_V12.csv'
voxel_size = 0.001

batch_calculate_cov_las(folder_path, output_csv_path, voxel_size)
