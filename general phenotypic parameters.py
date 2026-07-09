import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_canopy_metrics_voxel(las_path, voxel_size=0.01):
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T

    # 过滤无效点
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]

    if len(xyz) == 0:
        return {'CH': 0, 'CW': 0, 'CD': 0, 'PA': 0, 'CC': 0}

    # 冠层高度
    ch = xyz[:, 2].max() - xyz[:, 2].min()

    # 冠幅
    cw = xyz[:, 0].max() - xyz[:, 0].min()
    cd = xyz[:, 1].max() - xyz[:, 1].min()

    # xy 投影体素
    xy_points = xyz[:, :2]
    min_xy = xy_points.min(axis=0)
    voxel_coords = np.floor((xy_points - min_xy) / voxel_size).astype(int)
    unique_voxels = np.unique(voxel_coords, axis=0)
    num_occupied = unique_voxels.shape[0]

    # 地面投影面积
    pa = num_occupied * voxel_size * voxel_size

    # 总栅格数
    nx = int(np.ceil((xy_points[:,0].max() - min_xy[0]) / voxel_size))
    ny = int(np.ceil((xy_points[:,1].max() - min_xy[1]) / voxel_size))
    total_voxels = nx * ny

    cc = num_occupied / total_voxels if total_voxels > 0 else 0

    return {'CH': ch, 'CW': cw, 'CD': cd, 'PA': pa, 'CC': cc}

def batch_calculate_canopy_metrics_voxel(folder_path, output_csv_path, voxel_size=0.01):
    las_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.las')]
    results = []

    for las_file in tqdm(las_files, desc="计算冠层参数（体素投影）"):
        las_path = os.path.join(folder_path, las_file)
        metrics = calculate_canopy_metrics_voxel(las_path, voxel_size)
        metrics['LAS_File'] = las_file
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"所有 LAS 文件冠层参数计算完成，结果保存至 {output_csv_path}")


# ================== 使用示例 ==================
folder_path = r'I:\DJI_Workspace\V12-M350\temp'
output_csv_path = r'C:\Users\86188\Desktop\Canopy_V12.csv'
voxel_size = 0.01  # 体素边长，可根据点云密度调整

batch_calculate_canopy_metrics_voxel(folder_path, output_csv_path, voxel_size)
