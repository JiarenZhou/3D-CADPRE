import os
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline

# ================== 参数 ==================
HEIGHT_BIN = 0.01   # m，自适应 bin
MAX_HEIGHT = 2.3    # 固定纵轴范围
input_dir = r"I:\DJI_Workspace\V6-M350\final"
output_dir = r"I:\DJI_Workspace\V6-M350\vertical_profile"
os.makedirs(output_dir, exist_ok=True)

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# ================== 获取 LAS 文件 ==================
las_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".las")]
print(f"📦 发现 {len(las_files)} 个 LAS 文件")

# ================== 批量处理 ==================
for las_name in tqdm(las_files, desc="计算垂直点数分布"):
    las_path = os.path.join(input_dir, las_name)
    base = os.path.splitext(las_name)[0]

    las = laspy.read(las_path)
    z = las.z

    # ---------- 过滤异常点 ----------
    z = z[np.isfinite(z)]
    if len(z) == 0:
        continue

    # ---------- 局部坐标系 z ----------
    z_local = z - z.min()

    # ---------- 自适应 bin ----------
    bins = np.arange(0, MAX_HEIGHT + HEIGHT_BIN, HEIGHT_BIN)
    counts, edges = np.histogram(z_local, bins=bins)
    heights = (edges[:-1] + edges[1:]) / 2

    # ---------- 保存 CSV ----------
    df = pd.DataFrame({
        "Height_m": heights,
        "Point_Count": counts
    })
    csv_path = os.path.join(output_dir, f"{base}_vertical_point_count.csv")
    df.to_csv(csv_path, index=False)

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(4, 6))

    # 筛选非零散点
    mask_nonzero = counts > 0
    counts_nonzero = counts[mask_nonzero]
    heights_nonzero = heights[mask_nonzero]

    # 散点（仅非零点）
    ax.scatter(counts_nonzero, heights_nonzero, color='#1E90FF', s=10, alpha=0.7)

    # UnivariateSpline 拟合（仍使用全部高度，以保持曲线连续性）
    if len(heights) >= 5:
        spline = UnivariateSpline(heights, counts, s=len(counts) * 5)
        heights_smooth = np.linspace(0, MAX_HEIGHT, 300)
        counts_smooth = spline(heights_smooth)
        ax.plot(counts_smooth, heights_smooth, color='#FF00FF', linewidth=1.5)

    # ---------- 美化 ----------
    ax.set_xlabel("Point Count", fontsize=15)
    ax.set_ylabel("Height (m)", fontsize=15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1.2)
    ax.grid(False)
    ax.set_title("")

    # ---------- 固定坐标轴范围与步长 ----------
    ax.set_xlim(0, 2000)  # 横坐标固定范围
    ax.set_ylim(0, MAX_HEIGHT)  # 纵坐标固定范围
    ax.set_xticks(np.arange(0, 2001, 200))  # 横坐标刻度
    ax.set_yticks(np.arange(0, MAX_HEIGHT + 0.01, 0.2))  # 纵坐标刻度

    # ---------- 坐标刻度字体放大并倾斜 ----------
    plt.setp(ax.get_xticklabels(), fontsize=15, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontsize=15, rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{base}_vertical_point_count.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

print("🎉 所有点云垂直点数分布完成（固定高度，论文极简风格）")
