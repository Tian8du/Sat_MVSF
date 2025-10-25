
"""
This is a script for running the Sat-MVSF.
Copyright (C) <2023> <Jian Gao & GPCV>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from osgeo import gdal
import json
import numpy as np
import re
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import laspy
from tqdm import tqdm



def gdal_get_size(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    del dataset
    return width, height


def read_tfw(path):
    """
    TFW files are the ASCII files containing information
    for geocoding image data so TIFF images can be used
    directly by GIS and CAD applications.
    """
    file_object = open(path)
    try:
        all_the_text = file_object.read().splitlines()
    finally:
        file_object.close()

    tfw = np.array(all_the_text, dtype=np.float)

    if tfw.shape[0] != 6:
        raise Exception("6 parameters excepted in the tfw file, but got {}.".format(tfw.shape[0]))

    return tfw


def save_info_as_txt(txt_file, file_list):
    txt_str = str(len(file_list)) + "\n"

    for file, idx in zip(file_list, range(len(file_list))):
        txt_str += str(idx) + " " + file + "\n"

    with open(txt_file, "w") as f:
        f.write(txt_str)


def save_pair_as_txt(txt_file, len_file, ref_num):
    src_views = [i for i in range(len_file) if i != ref_num]

    txt_str = "1\n{}\n{}".format(ref_num, len(src_views))

    for sv in src_views:
        txt_str += " {} 99.99".format(sv)

    with open(txt_file, "w") as f:
        f.write(txt_str)


def save_border_as_txt(txt_file, border):
    txt_str = ""
    for b in border:
        txt_str += str(b) + "\n"

    with open(txt_file, "w") as f:
        f.write(txt_str)


def read_config(config_file):
    f = open(config_file, encoding='utf-8')
    config = json.load(f)
    f.close()
    return config


def save_json(json_file, dict_info):
    string = json.dumps(dict_info, indent=2)

    with open(json_file, 'w')as f:
        f.write(string)


def read_info_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    num = int(text[0])
    info_list = text[1:]

    assert num == len(info_list), "Error reading info file {}. It shows {} records, but get {} instead.".format(
        txt_file, num, len(info_list))

    read_info = dict()

    for info in info_list:
        items = info.split(" ", 1)
        read_info[int(items[0])] = items[1]
    
    return read_info

import os

def read_info_from_txt2(txt_file: str, make_absolute: bool = True, forward_slash: bool = True) -> dict[int, str]:
    """
    Read an info.txt file of format:
        N
        <id> <path>
        <id> <path>
    If paths are relative, convert them relative to txt_file's directory.

    Args:
        txt_file: Path to the .txt file
        make_absolute: If True, convert relative paths to absolute
        forward_slash: If True, replace backslashes with forward slashes

    Returns:
        Dictionary {id: resolved_path}
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        raise ValueError(f"Empty info file: {txt_file}")

    num = int(lines[0])
    info_list = lines[1:]
    assert num == len(info_list), (
        f"Error reading info file {txt_file}. It shows {num} records, but got {len(info_list)} instead."
    )

    base_dir = os.path.dirname(os.path.abspath(txt_file))
    result: dict[int, str] = {}

    for line in info_list:
        # Only split on first space; path can contain spaces
        idx_str, path_str = line.split(" ", 1)
        path_str = path_str.strip()

        if make_absolute and not os.path.isabs(path_str):
            path_str = os.path.normpath(os.path.join(base_dir, path_str))

        if forward_slash:
            path_str = path_str.replace("\\", "/")

        result[int(idx_str)] = path_str

    return result


def read_pair_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    num = int(text[0])

    assert num == int(len(text[1:])/2), "Error reading info file {}. It shows {} pairs, but get {} instead.".format(
        txt_file, num, int(len(text[1:])/2))

    view_info_list = []
    for i in range(num):
        view_info = [text[i * 2 + 1]]

        src_list = text[(i+1) * 2].split(" ")
        src_num = int(src_list[0])

        assert src_num == int(
            len(src_list[1:]) / 2), "Error reading info file {}. It shows {} sources, but get {} instead.".format(
            txt_file, src_num, int(len(src_list[1:]) / 2))

        for j in range(src_num):
            view_info.append(src_list[j*2+1])

        view_info_list.append(view_info)

    return view_info_list


def read_border_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    data = np.array(text, float)

    return data


def read_range_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    data = np.array(text, float)

    return data


# PFM file
def load_pfm(fname):
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0)
    return data


def save_pfm(file, image, scale=1):
    file = open(file, mode='wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(bytes('PF\n' if color else 'Pf\n', encoding='utf8'))
    file.write(bytes('%d %d\n' % (image.shape[1], image.shape[0]), encoding='utf8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(bytes('%f\n' % scale, encoding='utf8'))

    image_string = image.tostring()
    file.write(image_string)

    file.close()


# image
def read_img(filename):
    org = Image.open(filename)
    imgs = org.split()

    if len(imgs) == 3:
        img = org
    elif len(imgs) == 1:
        g = imgs[0]
        # img = Image.merge("RGB", (g, g, g))
        img = g
    else:
        raise Exception("Images must have 3 channels or 1.")

    return img


def gdal_read_img_pipeline(path, x_lu, y_lu, xsize, ysize):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    data = dataset.ReadAsArray(x_lu, y_lu, xsize, ysize)

    if len(data.shape) == 2:
        # cut off the small values
        below_thres = np.percentile(data.reshape((-1, 1)), 2)
        data[data < below_thres] = below_thres
        # cut off the big values
        above_thres = np.percentile(data.reshape((-1, 1)), 98)
        data[data > above_thres] = above_thres
        img = 255 * (data - below_thres) / (above_thres - below_thres)

        img = img.astype(np.uint8)
        data = np.stack([img, img, img], axis=0)

    del dataset

    return data


def cv_save_image(filepath, img):
    import cv2
    cv2.imwrite(filepath, img)


def gdal_create_raster(raster_path, width, height, nband, proj, geoTrans, invalid_value, dtype="Byte"):
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间

    if dtype == "Byte":
        raster_type = gdal.GDT_Byte
    elif dtype == "Float32":
        raster_type = gdal.GDT_Float32
    else:
        raise Exception("{} not supported yet.".format(dtype))

    dataset = driver.Create(raster_path, width, height, nband, raster_type)

    if invalid_value is not None:
        for i in range(nband):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(invalid_value)

    dataset.SetGeoTransform(geoTrans)
    # 左上角x坐标, 东西方向上图像的分辨率, 地图的旋转角度, 左上角y坐标, 地图的旋转角度, 南北方向上地图的分辨率)

    dataset.SetProjection(proj)

    text_ = str(geoTrans[1]) + "\n" + str(geoTrans[2]) + "\n" + str(geoTrans[4]) + "\n" + str(geoTrans[5]) + "\n"
    text_ += str(geoTrans[0] + float(geoTrans[1]) / 2) + "\n" + str(geoTrans[3] + float(geoTrans[5]) / 2)
    tfw_path = raster_path.replace(".tif", ".tfw")
    with open(tfw_path, "w") as f:
        f.write(text_)

    del driver, dataset


def gdal_write_to_tif(out_path, xlu, ylu, data):
    dataset = gdal.Open(out_path, gdal.GF_Write)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + out_path + " failed!")

    # 判读数组维数
    if data is None:
        return

    if len(data.shape) == 3:
        im_bands = data.shape[0]
    else:
        im_bands = 1

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data, xlu, ylu)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i], xlu, ylu)
    del dataset


def write_las(las_path, points):
    """
    使用 laspy 写 LAS/LAZ。
    points: (N,3) 的 numpy 数组，列为 [x, y, z]
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")

    # 选择点格式与版本（常用 1.2 / point_format 3，包含 XYZ+RGB 等；这里只用 XYZ 也可）
    header = laspy.LasHeader(point_format=3, version="1.2")

    # 合理设置 scale/offset，避免整数化精度损失或数值溢出
    # 这里用 0.01 米精度；offset 取各自最小值（常见做法）
    header.scales = (0.01, 0.01, 0.01)
    header.offsets = (float(points[:, 0].min()),
                      float(points[:, 1].min()),
                      float(points[:, 2].min()))

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 写 .las 或 .laz（如果路径以 .laz 结尾，需要安装 lazrs）
    las.write(las_path)


def read_las(las_path):
    """
    使用 laspy 一次性读取全部点（大文件会占内存）。
    若要低内存流式处理，请使用你已有的 rasterize_las_streaming_to_array。
    """
    las = laspy.read(las_path)  # 支持 .las /.laz
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64, copy=False)
    return points



def rasterize_las_streaming_to_array(
    las_path: str,
    x0: float, y0: float,
    xunit: float, yunit: float,
    nx: int, ny: int,
    reducer: str = "mean",
    nodata: float = -9999.0,
    chunk_points: int = 5_000_000,
):
    """
    Streaming point cloud rasterization (low-memory + real-time progress display).

    Args:
        las_path: Path to the LAS/LAZ file.
        x0, y0: Upper-left corner coordinates of the DSM grid.
        xunit, yunit: Grid resolution (pixel size) in x/y directions.
        nx, ny: Number of grid columns/rows.
        reducer: Aggregation method ("mean", "min", or "max").
        nodata: Value assigned to empty cells.
        chunk_points: Number of points processed per chunk (controls memory usage).

    Returns:
        out: Rasterized DSM as a float32 NumPy array.
    """
    try:
        import laspy
    except Exception as e:
        raise RuntimeError("laspy is required: pip install laspy tqdm") from e

    assert reducer in ("mean", "min", "max")
    nx, ny = int(nx), int(ny)
    shape = (ny, nx)

    # Initialize accumulation buffers
    if reducer == "mean":
        grid_sum = np.zeros(shape, dtype=np.float64)
        grid_cnt = np.zeros(shape, dtype=np.uint32)
        grid_agg = None
    else:
        grid_sum = None
        grid_cnt = None
        init_val = np.inf if reducer == "min" else -np.inf
        grid_agg = np.full(shape, init_val, dtype=np.float32)

    # Open LAS file and read points in chunks
    with laspy.open(las_path) as lasf:
        total_pts = lasf.header.point_count
        n_chunks = int(np.ceil(total_pts / chunk_points))
        print(f"   Start streaming rasterization: {las_path}")
        print(f"   Total points: {total_pts:,}  |  Chunks: {n_chunks} × {chunk_points}")

        pbar = tqdm(total=total_pts, unit="pts", dynamic_ncols=True)
        processed = 0

        for pts in lasf.chunk_iterator(chunk_points):
            # Convert ScaledArrayView → ndarray
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)

            # Compute grid indices
            col = np.floor((x - x0) / xunit).astype(np.int64)
            row = np.floor((y0 - y) / yunit).astype(np.int64)

            mask = (
                (col >= 0) & (col < nx) &
                (row >= 0) & (row < ny) &
                np.isfinite(z)
            )
            if not np.any(mask):
                pbar.update(len(x))
                continue

            rr, cc, zz = row[mask], col[mask], z[mask]

            # Accumulate or aggregate values
            if reducer == "mean":
                np.add.at(grid_sum, (rr, cc), zz)
                np.add.at(grid_cnt, (rr, cc), 1)
            elif reducer == "min":
                np.minimum.at(grid_agg, (rr, cc), zz)
            else:  # max
                np.maximum.at(grid_agg, (rr, cc), zz)

            processed += len(x)
            pbar.update(len(x))

        pbar.close()
        print(f"  Rasterization finished. Processed {processed:,} points.")

    # Finalize output DSM
    if reducer == "mean":
        out = np.full(shape, nodata, dtype=np.float32)
        valid = grid_cnt > 0
        out[valid] = (grid_sum[valid] / grid_cnt[valid]).astype(np.float32)
        return out
    else:
        out = grid_agg.astype(np.float32, copy=False)
        if reducer == "min":
            out[np.isinf(out)] = nodata
        else:
            out[np.isneginf(out)] = nodata
        return out

