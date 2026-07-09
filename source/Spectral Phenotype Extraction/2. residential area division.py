import os
import rasterio
import numpy as np

tif_path = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\seg_result_NIR.tif"
coord_path = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\coordinate_band.txt"
out_dir = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\plots\NIR"

os.makedirs(out_dir, exist_ok=True)

# 图像原点
origin_x = 0
origin_y = 0

with rasterio.open(tif_path) as src:

    img = src.read()
    profile = src.profile
    height = src.height
    width = src.width

    with open(coord_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):

        line = line.strip()
        if not line:
            continue

        p1, p2 = line.split()

        x1, y1 = map(float, p1.split(","))
        x2, y2 = map(float, p2.split(","))

        # 坐标平移 → 像素坐标
        col1 = int(x1 - origin_x)
        row1 = int(y1 - origin_y)

        col2 = int(x2 - origin_x)
        row2 = int(y2 - origin_y)

        # 排序避免反向
        rmin, rmax = sorted([row1, row2])
        cmin, cmax = sorted([col1, col2])

        # 边界保护
        rmin = max(0, rmin)
        cmin = max(0, cmin)
        rmax = min(height, rmax)
        cmax = min(width, cmax)

        if rmax <= rmin or cmax <= cmin:
            print(f"Skip {i}: invalid window")
            continue

        crop = img[:, rmin:rmax, cmin:cmax]

        new_profile = profile.copy()
        new_profile.update({
            "height": crop.shape[1],
            "width": crop.shape[2]
        })

        out_path = os.path.join(out_dir, f"{i}.tif")

        with rasterio.open(out_path, "w", **new_profile) as dst:
            dst.write(crop)

        print(f"Saved: {out_path}")