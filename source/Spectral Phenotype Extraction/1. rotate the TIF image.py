import os
import numpy as np
import rasterio
from scipy.ndimage import rotate
from rasterio.transform import Affine

# 输入文件夹和输出文件夹
input_folder = r"H:\2026公主岭无人机数据\M3T\V6-M3T\original"
output_folder = r"H:\2026公主岭无人机数据\M3T\V6-M3T\result"
os.makedirs(output_folder, exist_ok=True)

# 旋转角度（逆时针为正，单位：度）
angle = -17.49

def rotate_geotiff(input_tif, output_tif, angle):
    """旋转单个 GeoTIFF 并保留地理信息"""
    try:
        with rasterio.open(input_tif) as src:
            data = src.read()  # shape: (bands, height, width)
            profile = src.profile.copy()
            transform = src.transform
            width, height = src.width, src.height

        # 图像中心坐标
        cx_pix = width / 2
        cy_pix = height / 2
        cx_geo, cy_geo = transform * (cx_pix, cy_pix)  # 中心的地理坐标

        # 旋转每个波段（逆时针为正）
        rotated_bands = []
        for i in range(data.shape[0]):
            rotated_band = rotate(data[i], angle, reshape=True, order=1)  # 双线性插值
            rotated_bands.append(rotated_band)
        rotated_data = np.stack(rotated_bands)

        # 新图像大小
        new_height, new_width = rotated_data.shape[1], rotated_data.shape[2]

        # 计算新仿射变换，使中心点地理位置保持不变
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 原像素大小
        pixel_width = transform.a
        pixel_height = -transform.e

        # 构建新的仿射矩阵（旋转 + 平移）
        new_transform = Affine(
            cos_a * pixel_width, -sin_a * pixel_width, cx_geo - (cos_a*cx_pix - sin_a*cy_pix)*pixel_width,
            sin_a * pixel_height, cos_a * pixel_height, cy_geo - (sin_a*cx_pix + cos_a*cy_pix)*pixel_height
        )

        # 更新 profile
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

        # 保存旋转后的 GeoTIFF
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(rotated_data)

        print(f"完成: {os.path.basename(input_tif)}")
    except Exception as e:
        print(f"失败: {os.path.basename(input_tif)} -> {e}")

# 遍历文件夹批量处理
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".tif", ".tiff")):
        input_tif = os.path.join(input_folder, file_name)
        output_tif = os.path.join(output_folder, file_name)
        rotate_geotiff(input_tif, output_tif, angle)

print("所有文件处理完成！")
