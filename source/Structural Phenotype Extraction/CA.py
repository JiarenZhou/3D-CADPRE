import os
import csv
import numpy as np
import open3d as o3d

# ===================== 路径设置 =====================
MESH_DIR = r"G:\DJI_Workspace\V6-M350\surface"
OUTPUT_CSV = r"C:\Users\Jeron\Desktop\V6-Area.csv"
# ===================================================


def compute_mesh_surface_area(mesh):
    """
    计算三角网格的总表面积
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return 0.0

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    area = 0.5 * np.linalg.norm(
        np.cross(v1 - v0, v2 - v0),
        axis=1
    ).sum()

    return area


# ===================== 主流程 =====================
results = []

for fname in os.listdir(MESH_DIR):
    if not fname.lower().endswith(".ply"):
        continue

    fpath = os.path.join(MESH_DIR, fname)

    # 直接读取为 TriangleMesh
    mesh = o3d.io.read_triangle_mesh(fpath)

    if not mesh.has_triangles():
        print(f"Skip (no triangles): {fname}")
        continue

    surface_area = compute_mesh_surface_area(mesh)

    results.append([
        fname,
        round(surface_area, 6)
    ])

    print(f"{fname} | Surface Area = {surface_area:.6f} m²")


# ===================== 写 CSV =====================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Mesh_File",
        "Surface_Area_m2"
    ])
    writer.writerows(results)

print("Finished! Results saved to:", OUTPUT_CSV)
