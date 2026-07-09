import os
import numpy as np
import tifffile as tiff

# 输入输出路径
input_path = r"H:\2026公主岭无人机数据\M3T\V6-M3T\plots"
output_path = r"H:\2026公主岭无人机数据\M3T\V6-M3T\plots_final"

os.makedirs(output_path, exist_ok=True)

# 温度阈值
threshold = 35.0

print("开始处理")
files = os.listdir(input_path)

for file in files:
    if file.lower().endswith(".tif"):
        in_file = os.path.join(input_path, file)
        out_file = os.path.join(output_path, file)
        img = tiff.imread(in_file)
        mask = img < threshold
        img_masked = np.where(mask, img, np.nan)
        tiff.imwrite(out_file, img_masked.astype(np.float32))
        print(f"{file} 完成土壤剔除")

print("处理完毕")