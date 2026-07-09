import rasterio

# 读取TIF文件
tif_path = r'H:\2026公主岭无人机数据\M3T\V6-M3T\original\V6-M3T-12m.tif'
with rasterio.open(tif_path) as src:
    first_band = src.read(1)  # 获取第一个波段的值
    metadata = src.meta  # 获取原始TIF文件的元数据

# 保存第一个波段为新的TIF文件
output_path = r'H:\2026公主岭无人机数据\M3T\V6-M3T\original\V6-M3T-12m-band.tif'
metadata['count'] = 1  # 设置新文件的波段数
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(first_band, 1)  # 写入第一个波段的值
    print("成功提取波段")

