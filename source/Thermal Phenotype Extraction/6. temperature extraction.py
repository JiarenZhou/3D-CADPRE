import os
import numpy as np
import tifffile as tiff
import csv

input_path = r"H:\2026公主岭无人机数据\M3T\V6-M3T\plots_final"
output_csv = r"H:\2026公主岭无人机数据\M3T\V6-M3T\Temperature_metrics.csv"

print("开始计算温度参数")
results = []
files = os.listdir(input_path)

for file in files:
    if file.lower().endswith(".tif"):
        file_path = os.path.join(input_path, file)
        img = tiff.imread(file_path)
        img = np.squeeze(img)
        valid = img[~np.isnan(img)]
        if valid.size == 0:
            print(file, "no valid pixels")
            continue
        mean_temp = np.mean(valid)  # 平均温度
        std_temp = np.std(valid)    # 温度标准差
        min_temp = np.min(valid)    # 温度最小值
        max_temp = np.max(valid)    # 温度最大值
        cv = std_temp / mean_temp   # 冠层温度变异系数
        hot_threshold = 29.0        # 高温像元比例
        hot_ratio = np.sum(valid > hot_threshold) / valid.size
        Twet = np.percentile(valid, 5)   # 湿参考
        Tdry = np.percentile(valid, 95)  # 干参考
        if Tdry - Twet == 0:
            cwsi = np.nan
        else:
            cwsi = (mean_temp - Twet) / (Tdry - Twet)  # 作物水分胁迫指数
        cwsi = np.clip(cwsi, 0, 1)
        plot_name = file
        results.append([
            plot_name,
            mean_temp,
            std_temp,
            min_temp,
            max_temp,
            cv,
            hot_ratio,
            Twet,
            Tdry,
            cwsi
        ])
        print(f"{plot_name} done")

with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Plot",
        "Temp_mean",
        "Temp_std",
        "Temp_min",
        "Temp_max",
        "CV",
        "Hot_ratio",
        "Twet_5pct",
        "Tdry_95pct",
        "CWSI"
    ])
    writer.writerows(results)
print("保存至:", output_csv)