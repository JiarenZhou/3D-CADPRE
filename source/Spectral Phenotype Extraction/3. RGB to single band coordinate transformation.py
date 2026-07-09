input_txt = r"H:\2026公主岭无人机数据\M3M\V6-M3M-12m\coordinate_RGB.txt"
output_txt = r"H:\2026公主岭无人机数据\M3T\V6-M3T-12m\coordinate_band.txt"

# 旧坐标系
old_min_x, old_min_y = 0, 0
old_max_x, old_max_y = 8202,57679

# 新坐标系
new_min_x, new_min_y = 0, 0
new_max_x, new_max_y = 4832,33979

with open(input_txt, "r") as f:
    lines = f.readlines()

new_lines = []

for line in lines:

    line = line.strip()
    if not line:
        continue

    p1, p2 = line.split()
    x1_old, y1_old = map(float, p1.split(","))
    x2_old, y2_old = map(float, p2.split(","))

    # 线性映射到新坐标系
    x1_new = (x1_old - old_min_x) / (old_max_x - old_min_x) * (new_max_x - new_min_x) + new_min_x
    y1_new = (y1_old - old_min_y) / (old_max_y - old_min_y) * (new_max_y - new_min_y) + new_min_y

    x2_new = (x2_old - old_min_x) / (old_max_x - old_min_x) * (new_max_x - new_min_x) + new_min_x
    y2_new = (y2_old - old_min_y) / (old_max_y - old_min_y) * (new_max_y - new_min_y) + new_min_y

    new_line = f"{x1_new:.4f},{y1_new:.4f} {x2_new:.4f},{y2_new:.4f}\n"
    new_lines.append(new_line)

with open(output_txt, "w") as f:
    f.writelines(new_lines)

print("坐标转换完成，已保存：", output_txt)