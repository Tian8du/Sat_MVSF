
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


import os
import time
import glob
import numpy as np
import shutil
import copy
import cv2
from utils.io import gdal_get_size, gdal_read_img_pipeline, cv_save_image, gdal_create_raster, gdal_write_to_tif
from utils.io import gdal_write_to_tif, load_pfm, read_las, write_las
from utils.files import ensure_forward_slash
from utils.rpc_core import RPCModelParameter, load_rpc_as_array
from utils.projection import Projection
from predict import test
from modules.rpc_filter import filter_depth
from modules.generate_dsm import produce_dsm_from_points
from tqdm import tqdm
from utils.io import save_pfm


class Pipeline:
    def __init__(self, images_path, cameras_path, config, prj_info,
                 border_info, depth_range, workspace, logger, args):
        self.args = args
        self.logger = logger

        # data
        self.img_paths = images_path
        self.rpc_paths = cameras_path
        self.v_num = len(self.img_paths)

        # Img Size
        self.img_size = [gdal_get_size(path) for path in self.img_paths]

        self.depth_range = depth_range

        # prepare rpc
        self.rpcs = []
        idx = 0
        for path in self.rpc_paths:
            rpc = RPCModelParameter()
            rpc.load_from_file(path)
            rpc.Check_RPC(self.img_size[idx][0], self.img_size[idx][1], 100, 30)
            self.rpcs.append(copy.deepcopy(rpc))
            idx += 1
            
             # when the range is [0, 0, interval]
            if depth_range[0] == 0 and depth_range[1] == 0 and idx == 0:
                depth_range[1], depth_range[0] = rpc.GetH_MAX_MIN()

        # projection
        self.proj = Projection(prj_info)

        # load config
        self.run_crop_img = config["run_crop_img"]
        self.run_mvs = config["run_mvs"]
        self.run_generate_points = config["run_generate_points"]
        self.run_generate_dsm = config["run_generate_dsm"]
        self.run_generate_height_map = config["run_generate_height_map"]

        self.bxsize = config["block_size_x"]
        self.bysize = config["block_size_y"]
        self.xoverlap = 1 - config["overlap_x"]
        self.yoverlap = 1 - config["overlap_y"]
        self.para = config["para"]
        self.invalid = config["invalid_value"]

        self.p_thred = config["position_threshold"]
        self.d_thred = config["depth_threshold"]
        self.geo_num = config["geometric_num"]

        # prepare workspace
        self.out_path = workspace
        self.out_img_dir = ensure_forward_slash(os.path.join(self.out_path, "image"))
        self.out_rpc_dir = ensure_forward_slash(os.path.join(self.out_path, "rpc"))
        self.out_height_dir = ensure_forward_slash(os.path.join(self.out_path, "mvs"))
        self.out_las_dir = ensure_forward_slash(os.path.join(self.out_path, "points"))
        self.out_dsm_dir = ensure_forward_slash(os.path.join(self.out_path, "dsm"))


        self.out_img_paths = [ensure_forward_slash(os.path.join(self.out_img_dir, "{}".format(v))) for v in
                              range(self.v_num)]
        self.out_rpc_paths = [ensure_forward_slash(os.path.join(self.out_rpc_dir, "{}".format(v))) for v in
                              range(self.v_num)]
        self.out_height_paths = [ensure_forward_slash(os.path.join(self.out_height_dir, "{}".format(v))) for v in
                                 range(self.v_num)]

        self.xunit = border_info[4]
        self.yunit = border_info[5]
        self.border = border_info[:4]
        self.dsm_x_size = self.border[2]
        self.dsm_y_size = self.border[3]
        self.logger.info("border:{}".format(self.border))

        # Block info (Get from from_obj_img)

        if self.bxsize >= self.dsm_x_size or self.bysize >= self.dsm_y_size:
            self.block_process = False
            self.Jump = [1]

            self.img_bxsize, self.img_bysize = self.img_size[0][0], self.img_size[0][1]

            self.img_x_cen = np.array([[int(self.img_bxsize / 2)] for i in range(self.v_num)], int)
            self.img_y_cen = np.array([[int(self.img_bysize / 2)] for i in range(self.v_num)], int)

            self.x_grid_start = 0
            self.y_grid_start = 0

            self.x_dsm_start = [self.x_grid_start * self.xunit + self.border[0]]
            self.y_dsm_start = [- self.y_grid_start * self.yunit + self.border[1]]
            self.x_dsm_end = [self.dsm_x_size * self.xunit + self.border[0]]
            self.y_dsm_end = [- self.dsm_y_size * self.yunit + self.border[1]]

            self.b_num = 1

            self.bxsize = self.dsm_x_size
            self.bysize = self.dsm_y_size

        else:
            self.block_process = True

            self.Jump = np.ones(1)
            self.img_x_cen = np.zeros(1)
            self.img_y_cen = np.zeros(1)

            self.x_grid_start = np.zeros(1)
            self.y_grid_start = np.zeros(1)

            self.x_dsm_start = np.zeros(1)
            self.y_dsm_start = np.zeros(1)
            self.x_dsm_end = np.zeros(1)
            self.y_dsm_end = np.zeros(1)

            self.b_num = 0

            self.img_bxsize, self.img_bysize = self.from_obj_img()
            print("over")

    def create_folder(self):
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        if not os.path.exists(self.out_img_dir):
            os.mkdir(self.out_img_dir)
        if not os.path.exists(self.out_rpc_dir):
            os.mkdir(self.out_rpc_dir)
        if not os.path.exists(self.out_height_dir):
            os.mkdir(self.out_height_dir)
        if not os.path.exists(self.out_las_dir):
            os.mkdir(self.out_las_dir)
        if not os.path.exists(self.out_dsm_dir):
            os.mkdir(self.out_dsm_dir)

        for v in range(self.v_num):
            if not os.path.exists(self.out_img_paths[v]):
                os.mkdir(self.out_img_paths[v])

            if not os.path.exists(self.out_rpc_paths[v]):
                os.mkdir(self.out_rpc_paths[v])

            if not os.path.exists(self.out_height_paths[v]):
                os.mkdir(self.out_height_paths[v])

    def Set_Img_Bsize(self, img_bxsize, img_bysize):
        self.img_bxsize = img_bxsize
        self.img_bysize = img_bysize

        self.b_num = self.x_dsm_start.shape[0]

        img_x_start = (self.img_x_cen - self.img_bxsize / 2).astype(int)
        img_y_start = (self.img_y_cen - self.img_bysize / 2).astype(int)
        img_x_end = img_x_start + self.img_bxsize
        img_y_end = img_y_start + self.img_bysize

        # # Keep or remove?
        Jump_flag_list = []
        for v in range(self.v_num):
            Jump_flag = np.ones(self.b_num, dtype=int)

            index_x0_start0 = img_x_start[v] <= -self.xoverlap * self.img_bxsize # the next one will cover it, remove
            index_y0_start0 = img_y_start[v] <= -self.yoverlap * self.img_bysize

            index_x0_start1 = (img_x_start[v] < 0) & (img_x_start[v] > -self.xoverlap * self.img_bxsize)
            index_y0_start1 = (img_y_start[v] < 0) & (img_y_start[v] > -self.yoverlap * self.img_bysize)

            index_xX_start = img_x_start[v] > self.img_size[v][0] - 1  # Remove
            index_yY_start = img_y_start[v] > self.img_size[v][1] - 1  # Remove

            index_x0_end = img_x_end[v] < 0  # Remove
            index_y0_end = img_y_end[v] < 0  # Remove
            index_xX_end0 = img_x_end[v] >= self.img_size[v][0] - 1 + self.xoverlap * self.img_bxsize
            index_yY_end0 = img_y_end[v] >= self.img_size[v][1] - 1 + self.yoverlap * self.img_bysize

            index_xX_end1 = (img_x_end[v] > self.img_size[v][0] - 1) & (
                        img_x_end[v] < self.img_size[v][0] - 1 + self.xoverlap * self.img_bxsize)
            index_yY_end1 = (img_y_end[v] > self.img_size[v][1] - 1) & (
                        img_y_end[v] < self.img_size[v][1] - 1 + self.yoverlap * self.img_bysize)

            Jump_flag[index_xX_start] = 0
            Jump_flag[index_yY_start] = 0
            Jump_flag[index_x0_end] = 0
            Jump_flag[index_y0_end] = 0
            Jump_flag[index_x0_start0] = 0
            Jump_flag[index_y0_start0] = 0
            Jump_flag[index_xX_end0] = 0
            Jump_flag[index_yY_end0] = 0

            # If the Image block exceeds the border, the block is moved into the border
            img_x_start[v][index_x0_start1] = 0
            img_y_start[v][index_y0_start1] = 0
            img_x_end[v][index_x0_start1] = self.img_bxsize
            img_y_end[v][index_y0_start1] = self.img_bysize
            img_x_start[v][index_xX_end1] = self.img_size[v][0] - 1 - self.img_bxsize
            img_y_start[v][index_yY_end1] = self.img_size[v][1] - 1 - self.img_bysize
            img_x_end[v][index_xX_end1] = self.img_size[v][0] - 1
            img_y_end[v][index_yY_end1] = self.img_size[v][1] - 1

            # updata the img_cen_pts
            img_x_cen = (img_x_start + img_x_end) / 2
            img_y_cen = (img_y_start + img_y_end) / 2

            self.img_x_cen = img_x_cen.astype(int)
            self.img_y_cen = img_y_cen.astype(int)

            Jump_flag_list.append(Jump_flag)

        for flag in Jump_flag_list:
            self.Jump = self.Jump * flag

    def from_obj_img(self):
        BNumX = self.dsm_x_size / (self.bxsize * self.xoverlap)
        BNumY = self.dsm_y_size / (self.bysize * self.yoverlap)

        if abs(BNumX - int(BNumX)) < 0.00001:
            BNumX = int(BNumX)
        else:
            BNumX = int(BNumX + 1)
        if abs(BNumY - int(BNumY)) < 0.00001:
            BNumY = int(BNumY)
        else:
            BNumY = int(BNumY + 1)

        self.logger.info("BNumY: {} BNumX: {}".format(BNumY, BNumX))

        # Calculate the starting point of the DSM grid
        x_grid_start = (np.arange(BNumX) * self.xoverlap * self.bxsize).astype(int)
        y_grid_start = (np.arange(BNumY) * self.yoverlap * self.bysize).astype(int)

        x_grid_end = x_grid_start + self.bxsize
        y_grid_end = y_grid_start + self.bysize

        # If the DSM block exceeds the border, the block is moved into the border
        index_x = x_grid_end > self.dsm_x_size
        x_grid_start[index_x] = self.dsm_x_size - self.bxsize
        index_y = y_grid_end > self.dsm_y_size
        y_grid_start[index_y] = self.dsm_y_size - self.bysize

        x_grid_end = x_grid_start + self.bxsize
        y_grid_end = y_grid_start + self.bysize

        # meshgrid
        x_grid_start, y_grid_start = np.meshgrid(x_grid_start, y_grid_start)
        x_grid_end, y_grid_end = np.meshgrid(x_grid_end, y_grid_end)

        self.x_grid_start = x_grid_start.reshape(-1)
        self.y_grid_start = y_grid_start.reshape(-1)
        x_grid_end = x_grid_end.reshape(-1)
        y_grid_end = y_grid_end.reshape(-1)

        # grid -> projection coordinate
        self.x_dsm_start = self.x_grid_start * self.xunit + self.border[0]
        self.y_dsm_start = - self.y_grid_start * self.yunit + self.border[1]
        self.x_dsm_end = x_grid_end * self.xunit + self.border[0]
        self.y_dsm_end = - y_grid_end * self.yunit + self.border[1]

        proj_pts1 = np.stack((self.x_dsm_start, self.y_dsm_start), axis=-1)
        proj_pts2 = np.stack((self.x_dsm_end, self.y_dsm_end), axis=-1)
        proj_pts3 = np.stack((self.x_dsm_start, self.y_dsm_end), axis=-1)
        proj_pts4 = np.stack((self.x_dsm_end, self.y_dsm_start), axis=-1)

        proj_pts = np.stack((proj_pts1, proj_pts2, proj_pts3, proj_pts4), axis=0)

        # Geodetic coordinate
        # This is so wired!!!! why the long and lat change?
        geopts = self.proj.proj(proj_pts, reverse=True)

        lon_min = np.min(geopts[:, :, 0], axis=0)
        lon_max = np.max(geopts[:, :, 0], axis=0)
        lat_min = np.min(geopts[:, :, 1], axis=0)
        lat_max = np.max(geopts[:, :, 1], axis=0)

        temp = np.zeros(lat_min.shape, dtype=np.float64)

        img_x_start_pts_list = []
        img_y_start_pts_list = []
        img_x_end_pts_list = []
        img_y_end_pts_list = []
        img_x_bsize = []
        img_y_bsize = []

        for rpc in self.rpcs:
            h_min = temp + self.depth_range[0]
            h_max = temp + self.depth_range[1]

            geopts1 = np.stack((lat_min, lon_min, h_min), axis=-1)
            geopts2 = np.stack((lat_min, lon_max, h_min), axis=-1)
            geopts3 = np.stack((lat_max, lon_min, h_min), axis=-1)
            geopts4 = np.stack((lat_max, lon_max, h_min), axis=-1)
            geopts5 = np.stack((lat_min, lon_min, h_max), axis=-1)
            geopts6 = np.stack((lat_min, lon_max, h_max), axis=-1)
            geopts7 = np.stack((lat_max, lon_min, h_max), axis=-1)
            geopts8 = np.stack((lat_max, lon_max, h_max), axis=-1)

            geopts = np.stack((geopts1, geopts2, geopts3, geopts4, geopts5, geopts6, geopts7, geopts8), axis=0)
            geopts = geopts.reshape(-1, 3)

            samp, line = rpc.RPC_OBJ2PHOTO(geopts[:, 0], geopts[:, 1], geopts[:, 2])
            samp = samp.reshape((8, -1))
            line = line.reshape((8, -1))

            img_x_start_pts = np.min(samp, axis=0)
            img_y_start_pts = np.min(line, axis=0)
            img_x_end_pts = np.max(samp, axis=0)
            img_y_end_pts = np.max(line, axis=0)

            img_x_start_pts_list.append(img_x_start_pts)
            img_y_start_pts_list.append(img_y_start_pts)
            img_x_end_pts_list.append(img_x_end_pts)
            img_y_end_pts_list.append(img_y_end_pts)

            temp_xsize = np.max(img_x_end_pts - img_x_start_pts)
            temp_ysize = np.max(img_y_end_pts - img_y_start_pts)

            # Enables image blocks to be divisible by  para
            # temp_xsize = int(temp_xsize / self.para + 0.5) * self.para
            # temp_ysize = int(temp_ysize / self.para + 0.5) * self.para
            temp_xsize = int(temp_xsize / self.para + 1) * self.para
            temp_ysize = int(temp_ysize / self.para + 1) * self.para

            img_x_bsize.append(temp_xsize)
            img_y_bsize.append(temp_ysize)

        img_x_start_pts_list = np.array(img_x_start_pts_list)
        img_y_start_pts_list = np.array(img_y_start_pts_list)
        img_x_end_pts_list = np.array(img_x_end_pts_list)
        img_y_end_pts_list = np.array(img_y_end_pts_list)

        img_x_cen = (img_x_start_pts_list + img_x_end_pts_list) / 2
        img_y_cen = (img_y_start_pts_list + img_y_end_pts_list) / 2

        # The center of the image block
        self.img_x_cen = img_x_cen.astype(int)
        self.img_y_cen = img_y_cen.astype(int)

        img_x_bsize = np.max(img_x_bsize)
        img_y_bsize = np.max(img_y_bsize)

        self.Set_Img_Bsize(img_x_bsize, img_y_bsize)

        return img_x_bsize, img_y_bsize

    def crop_image(self, i):
        if self.Jump[i] == 0:
            return

        for v in range(self.v_num):
            out_name = "block{:0>4d}".format(i)
            img_x_start = self.img_x_cen[v][i] - int(self.img_bxsize / 2)
            img_y_start = self.img_y_cen[v][i] - int(self.img_bysize / 2)

            img = gdal_read_img_pipeline(self.img_paths[v], int(img_x_start), int(img_y_start), int(self.img_bxsize),
                                         int(self.img_bysize))
            img = img.transpose([1, 2, 0])
            
            out_path = os.path.join(self.out_img_paths[v], "{}.png".format(out_name)).replace("\\", "/")
            cv_save_image(out_path, img)

            rpc = self.rpcs[v]
            block_rpc = copy.deepcopy(rpc)
            block_rpc.SAMP_OFF -= int(img_x_start)
            block_rpc.LINE_OFF -= int(img_y_start)
            
            out_path = os.path.join(self.out_rpc_paths[v], "{}.rpc".format(out_name)).replace("\\", "/")
            block_rpc.save_dirpc_to_file(out_path)

    def generate_points(self, i):
        if self.Jump[i] == 0:
            return

        out_name = "block{:0>4d}".format(i)

        las_path = os.path.join(self.out_las_dir, out_name+".las").replace("\\", "/")
        if os.path.exists(las_path):
            os.remove(las_path)

        # filter heights
        heights = []
        rpcs = []

        view = [i for i in range(self.v_num)]


        for v in view:
            height_map_path = os.path.join(self.out_height_paths[v], "init/{}.pfm".format(
                out_name)).replace("\\", "/")
            height_map = load_pfm(height_map_path)
            heights.append(height_map)

            # 读取RPC
            rpc_path = os.path.join(self.out_rpc_paths[v], "{}.rpc".format(out_name)).replace("\\", "/")
            rpc, _, _ = load_rpc_as_array(rpc_path)
            rpcs.append(rpc)

        heights = np.stack(heights, axis=0)
        rpcs = np.stack(rpcs, axis=0)

        mask, height_est_averaged = filter_depth(heights, rpcs, p_ratio=self.p_thred, d_ratio=self.d_thred,
                                                 geo_consist_num=self.geo_num, prob=None, confidence_ratio=0.2)


        height_est_averaged = height_est_averaged.reshape(-1)
        mask = mask.reshape(-1)

        x = np.arange(0.0, self.img_bxsize, 1.0)
        y = np.arange(0.0, self.img_bysize, 1.0)

        x, y = np.meshgrid(x, y)
        x = x.reshape(-1)
        y = y.reshape(-1)

        height_map = height_est_averaged[mask]
        x = x[mask]
        y = y[mask]

        ref_rpc = RPCModelParameter(rpcs[0])
        lat, lon = ref_rpc.RPC_PHOTO2OBJ(x, y, height_map)
        geopts = np.stack((lon, lat), axis=-1)

        if len(geopts) == 0:
            return None
        # print("{}/{} finished RPC_PHOTO2OBJ".format(i, self.b_num))
        # geopts -> proj_pts
        projpts = self.proj.proj(geopts, False)
        # print("{}/{} finished projection".format(i, self.b_num))

        points = np.stack((projpts[:, 0], projpts[:, 1], height_map), axis=-1)
        
        return points
        # save as point cloud
        # write_las(las_path, points)

    def generate_height_map_masked(self, i):
        if self.Jump[i] == 0:
            return

        out_name = "block{:0>4d}".format(i)

        # filter heights
        heights = []
        rpcs = []

        view = [i for i in range(self.v_num)]

        for v in view:
            height_map_path = os.path.join(self.out_height_paths[v], "init/{}.pfm".format(
                out_name)).replace("\\", "/")
            height_map = load_pfm(height_map_path)
            heights.append(height_map)

            # 读取RPC
            rpc_path = os.path.join(self.out_rpc_paths[v], "{}.rpc".format(out_name)).replace("\\", "/")
            rpc, _, _ = load_rpc_as_array(rpc_path)
            rpcs.append(rpc)
        depth_mask_path =  os.path.join(self.out_height_paths[0], "depth_masked/{}.pfm".format(
                out_name)).replace("\\", "/")
        heights = np.stack(heights, axis=0)
        rpcs = np.stack(rpcs, axis=0)

        mask, height_est_averaged = filter_depth(heights, rpcs, p_ratio=self.p_thred, d_ratio=self.d_thred,
                                                 geo_consist_num=self.geo_num, prob=None, confidence_ratio=0.2)
        depth_to_save = np.where(mask, height_est_averaged, -999.0).astype(np.float32)
        save_pfm(depth_mask_path, depth_to_save)
        save_pfm(depth_mask_path,height_est_averaged)

        return None

    # def merge_points(self):
    #     las_lists = glob.glob("{}/*.las".format(self.out_las_dir))
    #     las_lists = [dl.replace("\\", "/") for dl in las_lists]

    #     points = []
    #     self.logger.info("merging points: ")
    #     with tqdm(total=self.b_num) as pbar:
    #         pbar.set_description("merging points: ")
    #         for las_file in las_lists:
    #             points.append(read_las(las_file))
    #             # os.remove(las_file)
    #             pbar.update(1)

    #     points = np.concatenate(points, axis=0)

    #     write_las(ensure_forward_slash(os.path.join(self.out_las_dir, "points.las")), points)

    def generate_dsm(self):
        dsm_path = os.path.join(self.out_dsm_dir, "dsm.tif").replace("\\", "/")

        if os.path.exists(dsm_path):
            os.remove(dsm_path)

        geo_trans = [self.border[0] - float(self.xunit) / 2, self.xunit, 0,
                     self.border[1] - float(-self.yunit) / 2, 0, -self.yunit]
        gdal_create_raster(dsm_path, int(self.dsm_x_size), int(self.dsm_y_size), 1,
                           self.proj.spatial_reference.ExportToWkt(), geo_trans,
                           self.invalid, dtype="Float32")

        points = read_las(ensure_forward_slash(os.path.join(self.out_las_dir, "points.las")))

        dsm = produce_dsm_from_points(points, self.border[0], self.border[1], self.xunit,
                                      self.yunit, int(self.dsm_x_size), int(self.dsm_y_size))

        gdal_write_to_tif(dsm_path, 0, 0, dsm)

        import matplotlib.pyplot as plt
        dsm = cv2.imread(dsm_path, cv2.IMREAD_ANYDEPTH)
        dsm[dsm <= self.invalid] = np.nan
        dsm[np.isnan(dsm)] = np.nanmin(dsm) - 1
        plt.imsave(dsm_path.replace(".tif", ".jpg"), dsm)

    def run(self):
        self.logger.info("RPC Pipeline")
        total_start = time.time()

        self.create_folder()
        if self.run_crop_img:
            start = time.time()

            with tqdm(total=self.b_num) as pbar:
                pbar.set_description("Crop Image: ")
                for i in tqdm(range(self.b_num)):
                    self.crop_image(i)
                    pbar.update(1)

            end = time.time()
            self.logger.info("[Crop Image] Cost {:.4f} min".format((end - start) / 60.0))
        else:
            self.logger.info("[Crop Image] Skip!")

        if self.run_mvs:
            start = time.time()

            test(self.out_path, self.depth_range, self.v_num, self.args, self.logger)

            end = time.time()
            self.logger.info("[MVS] Cost {:.4f} min".format((end - start) / 60.0))
        else:
            self.logger.info("[MVS] Skip!")

        if self.run_generate_points:
            start = time.time()
            
            points = []
            with tqdm(total=self.b_num) as pbar:
                pbar.set_description("Generate Points: ")
                for i in tqdm(range(self.b_num)):
                    pts = self.generate_points(i)
                    if pts is not None:    
                        # print(i, len(pts))
                        points.append(pts)
                    pbar.update(1)

            points = np.concatenate(points, axis=0)
            write_las(ensure_forward_slash(os.path.join(self.out_las_dir, "points.las")), points)

            end = time.time()
            self.logger.info("[Generate points] Cost {:.4f} min".format((end - start) / 60.0))
        else:
            self.logger.info("[Generate points] Skip!")

        if self.run_generate_height_map:
            start = time.time()
            with tqdm(total=self.b_num) as pbar:
                pbar.set_description("Generate Points: ")
                for i in tqdm(range(self.b_num)):
                    self.generate_height_map_masked(i)
                    pbar.update(1)
            end = time.time()
            self.logger.info("[Generate masked height map] Cost {:.4f} min".format((end - start) / 60.0))
        else:
            self.logger.info("[Generate masked height map] Skip!")

        if self.run_generate_dsm:
            start = time.time()
            self.generate_dsm()
            end = time.time()
            self.logger.info("[Generate DSM] Cost {:.4f} min".format((end - start) / 60.0))
            self.logger.info("Mosaic DSM Finished!")
        else:
            self.logger.info("[Generate DSM] Skip!")

        total_end = time.time()
        self.logger.info("Totally Cost {} min".format(
            (total_end - total_start) / 60.0))


if __name__ == "__main__":
    pass
