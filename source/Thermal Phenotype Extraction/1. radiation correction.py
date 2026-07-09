# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import subprocess
from pyexiv2 import Image

# 参数设置
tsdk = r"E:\DJI Thermal SDK\release_x64\dji_irp.exe"                    # dji_irp.exe的路径
path = r"H:\DJI_202606121705_009_gongzhuling-M3T-20260612-15m(2)"    # M3T采集影像的路径，不要有中文
savepath = r"H:\V6-M3T-15m"                                                 # 结果路径，不要有中文
os.makedirs(savepath, exist_ok=True)

distance = 15  # 飞行高度
distance = min(distance, 25)  # SDK最大支持25m
emissivity = 0.95  # 反射率，不变
humidity = 40  # 环境湿度，查询得到

# 三块标定板温度（标定板非必须，也可用平均土壤温度代替）
# black_temp = 74.15
# gray_temp = 60.95
# white_temp = 48.45
# reflection = (black_temp + gray_temp + white_temp) / 3  # 环境背景温度，用于标定
reflection = 24.89

# 1. 使用 DJI Thermal SDK 提取 RAW 温度信息
def use_tsdk(tsdk, path, savepath, reflection):
    print("========== 开始提取热红外RAW数据 ==========")
    imgnamelist = os.listdir(path)
    for imgname in imgnamelist:
        if imgname.lower().endswith(".jpg") and "T" in imgname:
            coreimgname = os.path.splitext(imgname)[0]
            input_file = os.path.join(path, imgname)
            output_file = os.path.join(savepath, coreimgname + ".raw")
            cmd = [
                tsdk,
                "-s", input_file,
                "-o", output_file,
                "-a", "measure",
                "--distance", str(distance),
                "--humidity", str(humidity),
                "--emissivity", str(emissivity),
                "--reflection", str(reflection)
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print(f"处理成功：{imgname}")
            else:
                print(f"处理失败：{imgname}，返回码 {result.returncode}")
    print("========== RAW提取完成 ==========\n")


# 2. RAW 转换为 TIF 温度图，备用
def raw_to_tif(path, rows, cols):
    print("========== 开始 RAW 转换为 TIFF ==========")
    files = os.listdir(path)
    for file in files:
        portion = os.path.splitext(file)
        if portion[1].lower() == ".raw":
            raw_path = os.path.join(path, file)
            try:
                img = np.fromfile(raw_path, dtype="int16")
                img = img / 10.0
                img = img.reshape(rows, cols)
                tif_name = portion[0] + ".tif"
                tif_path = os.path.join(path, tif_name)
                cv2.imwrite(
                    tif_path,
                    img,
                    (int(cv2.IMWRITE_TIFF_COMPRESSION), 1)
                )
                print(f"转换完成：{tif_name}")
            except Exception as e:
                print(f"转换失败：{file}，原因：{e}")
    print("========== TIFF转换完成 ==========\n")


# 3. 复制 EXIF 信息
def exifrw(path, exif_path):
    print("========== 开始写入EXIF信息 ==========")
    unprocessed = 0
    tif_files = os.listdir(path)
    rjpeg_files = os.listdir(exif_path)
    for read_file in rjpeg_files:
        if "T" in read_file:
            portion = os.path.splitext(read_file)
            tif_name = portion[0] + ".tif"
            if tif_name in tif_files:
                try:
                    tif_path = os.path.join(path, tif_name)
                    rjpeg_path = os.path.join(exif_path, read_file)
                    img = Image(tif_path)
                    rjpeg = Image(rjpeg_path)
                    exif = rjpeg.read_exif()
                    img.modify_exif(exif)
                    rjpeg.close()
                    img.close()
                    print(f"EXIF写入成功：{tif_name}")
                except Exception as e:
                    print(f"EXIF写入失败：{tif_name}，原因：{e}")
                    unprocessed += 1
            else:
                print(f"未找到对应TIFF文件：{tif_name}")
                unprocessed += 1
    print("========== EXIF写入完成 ==========")
    print(f"未处理文件数量：{unprocessed}")
    print("========== 全部处理结束 ==========")


# 主程序
if __name__ == "__main__":
    use_tsdk(tsdk, path, savepath, reflection)
    raw_to_tif(savepath, 512, 640)
    exifrw(savepath, path)