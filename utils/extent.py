from osgeo import gdal
from pyproj import Transformer, CRS
import os

from osgeo import gdal
from pyproj import Transformer, CRS
import os

def generate_dsm_info_files(img_f, img_n, img_b, resolution=2.5, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    def get_rpccorners_latlon(image_path):
        ds = gdal.Open(image_path)
        rpc = ds.GetMetadata('RPC')
        if rpc is None:
            raise ValueError("No RPC info found in this dataset.")
        rpc_transformer = gdal.Transformer(ds, None, ["METHOD=RPC"])
        if rpc_transformer is None:
            raise ValueError("Cannot create RPC transformer.")
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        corners_pix = [(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)]
        lat_list, lon_list = [], []
        for x_pix, y_pix in corners_pix:
            success, xyz = rpc_transformer.TransformPoint(0, x_pix, y_pix, 0)
            if not success:
                raise RuntimeError(f"RPC Transform failed for point {x_pix},{y_pix}")
            lon, lat = xyz[0], xyz[1]
            lon_list.append(lon)
            lat_list.append(lat)
        return min(lat_list), max(lat_list), min(lon_list), max(lon_list)

    # 提取重叠区域
    min_lat_f, max_lat_f, min_lon_f, max_lon_f = get_rpccorners_latlon(img_f)
    min_lat_n, max_lat_n, min_lon_n, max_lon_n = get_rpccorners_latlon(img_n)
    min_lat_b, max_lat_b, min_lon_b, max_lon_b = get_rpccorners_latlon(img_b)

    overlap_min_lat = max(min_lat_f, min_lat_n, min_lat_b)
    overlap_max_lat = min(max_lat_f, max_lat_n, max_lat_b)
    overlap_min_lon = max(min_lon_f, min_lon_n, min_lon_b)
    overlap_max_lon = min(max_lon_f, max_lon_n, max_lon_b)

    center_lat = (overlap_min_lat + overlap_max_lat) / 2
    center_lon = (overlap_min_lon + overlap_max_lon) / 2
    zone_number = int((center_lon + 180) / 6) + 1
    epsg_code = 32600 + zone_number if center_lat >= 0 else 32700 + zone_number

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x_min, y_min = transformer.transform(overlap_min_lon, overlap_min_lat)
    x_max, y_max = transformer.transform(overlap_max_lon, overlap_max_lat)

    cols = int((x_max - x_min) / resolution)
    rows = int((y_max - y_min) / resolution)

    # 1️⃣ Save projection.prj
    crs = CRS.from_epsg(epsg_code)
    prj_wkt = crs.to_wkt()
    with open(os.path.join(output_dir, "projection.prj"), "w", encoding="utf-8") as f:
        f.write(prj_wkt)
    print("Saved projection.prj")

    # 2️⃣ Save border.txt
    with open(os.path.join(output_dir, "border.txt"), "w", encoding="utf-8") as f:
        f.write(f"{x_min}\n{y_max}\n{cols}\n{rows}\n{resolution}\n{resolution}\n")
    print("Saved border.txt")

    # 3️⃣ Save cameras_info.txt
    def rpc_path(img_path):
        return img_path.replace(".tif", "_ba.rpc")

    cameras_info = [
        "3",
        f"0 {rpc_path(img_f)}",
        f"1 {rpc_path(img_n)}",
        f"2 {rpc_path(img_b)}"
    ]
    with open(os.path.join(output_dir, "cameras_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cameras_info))
    print("Saved cameras_info.txt")

    # 4️⃣ Save images_info.txt
    images_info = [
        "3",
        f"0 {img_f}",
        f"1 {img_n}",
        f"2 {img_b}"
    ]
    with open(os.path.join(output_dir, "images_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(images_info))
    print("Saved images_info.txt")

    # 5️⃣ Save pair.txt
    pair_info = [
        "1",
        "1",
        "2 0 0.9 2 0.9"
    ]
    with open(os.path.join(output_dir, "pair.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pair_info))
    print("Saved pair.txt")

    # 6️⃣ Save range.txt
    range_info = [
        "0",
        "0",
        str(resolution)
    ]
    with open(os.path.join(output_dir, "range.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(range_info))
    print("Saved range.txt")

    # 输出可用 gdalwarp 命令
    print("\n✅ All DSM info files generated successfully.\n")
    print("Use with gdalwarp:")
    print(f"gdalwarp -t_srs EPSG:{epsg_code} -te {x_min} {y_min} {x_max} {y_max} -tr {resolution} {resolution} input.tif output_dsm.tif")

# ===== 使用示例 =====
# generate_dsm_info_files(img_f, img_n, img_b, resolution=2.5, output_dir="./dsm_info")


def get_rpccorners_latlon(image_path):
    ds = gdal.Open(image_path)
    rpc = ds.GetMetadata('RPC')
    if rpc is None:
        raise ValueError("No RPC info found in this dataset.")
    rpc_transformer = gdal.Transformer(ds, None, ["METHOD=RPC"])
    if rpc_transformer is None:
        raise ValueError("Cannot create RPC transformer.")
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    corners_pix = [(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)]
    lat_list, lon_list = [], []
    for x_pix, y_pix in corners_pix:
        success, xyz = rpc_transformer.TransformPoint(0, x_pix, y_pix, 0)
        if not success:
            raise RuntimeError(f"RPC Transform failed for point {x_pix},{y_pix}")
        lon, lat = xyz[0], xyz[1]
        lon_list.append(lon)
        lat_list.append(lat)
    return min(lat_list), max(lat_list), min(lon_list), max(lon_list)

def process_zy3_extent(img_f, img_n, img_b, resolution=2.5, output_dir="."):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取重叠范围（WGS84）
    min_lat_f, max_lat_f, min_lon_f, max_lon_f = get_rpccorners_latlon(img_f)
    min_lat_n, max_lat_n, min_lon_n, max_lon_n = get_rpccorners_latlon(img_n)
    min_lat_b, max_lat_b, min_lon_b, max_lon_b = get_rpccorners_latlon(img_b)

    overlap_min_lat = max(min_lat_f, min_lat_n, min_lat_b)
    overlap_max_lat = min(max_lat_f, max_lat_n, max_lat_b)
    overlap_min_lon = max(min_lon_f, min_lon_n, min_lon_b)
    overlap_max_lon = min(max_lon_f, max_lon_n, max_lon_b)

    print("\n*** ZY-3 Three-View Overlap in WGS84 ***")
    print(f"Lat: {overlap_min_lat:.6f} ~ {overlap_max_lat:.6f}")
    print(f"Lon: {overlap_min_lon:.6f} ~ {overlap_max_lon:.6f}")

    # 自动选取 UTM 分带 EPSG
    center_lat = (overlap_min_lat + overlap_max_lat) / 2
    center_lon = (overlap_min_lon + overlap_max_lon) / 2
    zone_number = int((center_lon + 180) / 6) + 1
    epsg_code = 32600 + zone_number if center_lat >= 0 else 32700 + zone_number
    print(f"\nUsing UTM Zone {zone_number} (EPSG:{epsg_code})")

    # 经纬度 -> 投影坐标范围
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x_min, y_min = transformer.transform(overlap_min_lon, overlap_min_lat)
    x_max, y_max = transformer.transform(overlap_max_lon, overlap_max_lat)

    print("\n*** Overlap in Projected Coordinates ***")
    print(f"X range (East): {x_min:.3f} ~ {x_max:.3f} meters")
    print(f"Y range (North): {y_min:.3f} ~ {y_max:.3f} meters")

    cols = int((x_max - x_min) / resolution)
    rows = int((y_max - y_min) / resolution)

    print(f"\nDSM Output Size: {cols} cols x {rows} rows @ {resolution} m/pixel")

    # 保存 border.txt
    x_topleft = x_min
    y_topleft = y_max
    border_path = os.path.join(output_dir, "border.txt")
    with open(border_path, "w", encoding="utf-8") as f:
        f.write(f"{x_topleft}\n{y_topleft}\n{cols}\n{rows}\n{resolution}\n{resolution}\n")
    print(f"Saved {border_path}")

    # 保存 projection.prj
    crs = CRS.from_epsg(epsg_code)
    prj_wkt = crs.to_wkt()
    prj_path = os.path.join(output_dir, "projection.prj")
    with open(prj_path, "w", encoding="utf-8") as f:
        f.write(prj_wkt)
    print(f"Saved {prj_path}")

    # 输出 gdalwarp 命令建议
    print("\nUse with gdalwarp:")
    print(f"gdalwarp -t_srs EPSG:{epsg_code} -te {x_min} {y_min} {x_max} {y_max} -tr {resolution} {resolution} input.tif output_dsm.tif")

# 使用示例（用户替换为真实路径）
if __name__ == "__main__":
    img_f = r"H:/data/ZY3-Data/ZY3MVS/ZY3_01a_mynbavp_278116_20140827_183905_0008_SASMAC_CHN_sec_rel_001_14082905653.tif"
    img_n = r"H:/data/ZY3-Data/ZY3MVS/ZY3_01a_mynnavp_278116_20140827_183835_0007_SASMAC_CHN_sec_rel_001_14082905583.tif"
    img_b = r"H:/data/ZY3-Data/ZY3MVS/ZY3_01a_mynfavp_278116_20140827_183807_0008_SASMAC_CHN_sec_rel_001_14082905709.tif"
    # process_zy3_extent(img_f, img_n, img_b, resolution=2.5, output_dir="./output")
    generate_dsm_info_files(img_f, img_n, img_b, resolution=2.5, output_dir="./dsm_info")
