
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


from torch.utils.data import Dataset
from utils.io import *
from dataset.preprocess import *
from dataset.gen_list import gen_all_mvs_list_rpc, gen_ref_list_rpc
from imageio import imread
from utils.rpc_core import load_rpc_as_array
import os


class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, depth_range, args):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.view_num = view_num
        self.depth_range = depth_range
        self.args = args
        assert self.mode in ["train", "val", "test", "pred"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        if self.mode == "pred":
            sample_list = gen_all_mvs_list_rpc(self.data_folder, self.view_num)
        else:
            sample_list = gen_ref_list_rpc(self.data_folder, self.view_num, 2)

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    @ staticmethod
    def read_depth(filename):
        # read depth file
        if os.path.splitext(filename)[1] == ".pfm":
            depth_image = np.float32(load_pfm(filename))
        elif os.path.splitext(filename)[1] == ".tif":
            depimg = imread(filename)
            depth_image = np.float32(depimg)
        else:
            depth_image = None

        return np.array(depth_image)

    def get_sample(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######
        outimage = None
        outrpc = None

        centered_images = []
        rpc_paramters = []
        croped_depth = None
        depth_min = None
        depth_max = None

        # depth
        depth_image = self.read_depth(os.path.join(data[2 * self.view_num]))

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(data[2 * view]))
            else:
                image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            rpc, d_max, d_min = load_rpc_as_array(data[2 * view + 1])

            if view == 0:
                # determine a proper scale to resize input
                scaled_image, scaled_rpc, scaled_depth = scale_input_rpc(image, rpc, depth_image=depth_image,
                                                                     scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_rpc, croped_depth = crop_input_rpc(scaled_image, scaled_rpc, depth_image=scaled_depth,
                                                                    max_h=self.args.max_h, max_w=self.args.max_w,
                                                                    resize_scale=self.args.resize_scale)
                outimage = croped_image
                outrpc = croped_rpc
                depth_min = d_min
                depth_max = d_max
            else:
                # determine a proper scale to resize input
                scaled_image, scaled_rpc, _ = scale_input_rpc(image, rpc, scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_rpc, _ = crop_input_rpc(scaled_image, scaled_rpc, max_h=384, max_w=768, resize_scale=1.0)

            # scale cameras for building cost volume
            scaled_rpc = scale_rpc(croped_rpc, scale=1.0)
            # multiply intrinsics and extrinsics to get projection matrix
            rpc_paramters.append(scaled_rpc)
            centered_images.append(center_image(croped_image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.stack(rpc_paramters)

        # Depth
        if self.depth_range is not None:
            depth_min = self.depth_range[0]
            depth_max = self.depth_range[1]

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        mask = np.float32((croped_depth >= depth_min) * 1.0) * np.float32((croped_depth <= depth_max) * 1.0)

        h, w = croped_depth.shape
        depth_ms = {
            "stage1": cv2.resize(croped_depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(croped_depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": croped_depth
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] = stage3_rpc[:, 0] / 4
        stage3_rpc[:, 1] = stage3_rpc[:, 1] / 4
        stage3_rpc[:, 5] = stage3_rpc[:, 5] / 4
        stage3_rpc[:, 6] = stage3_rpc[:, 6] / 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outrpc,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_pred_sample(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######
        outimage = None
        outrpc = None

        centered_images = []
        rpc_paramters = []
        depth_min = None
        depth_max = None

        for view in range(self.view_num):
            # Images
            image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            rpc, d_max, d_min = load_rpc_as_array(data[2 * view + 1])

            if view == 0:
                # determine a proper scale to resize input
                scaled_image, scaled_rpc, scaled_depth = scale_input_rpc(image, rpc, depth_image=None,
                                                                     scale=self.args.resize_scale)
                outimage = scaled_image
                outrpc = scaled_rpc
                depth_min = d_min
                depth_max = d_max
            else:
                # determine a proper scale to resize input
                scaled_image, scaled_rpc, _ = scale_input_rpc(image, rpc, scale=self.args.resize_scale)

            # scale cameras for building cost volume
            scaled_rpc = scale_rpc(scaled_rpc, scale=1.0)
            # multiply intrinsics and extrinsics to get projection matrix
            rpc_paramters.append(scaled_rpc)
            centered_images.append(center_image(scaled_image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.stack(rpc_paramters)

        # Depth
        # print(new_ndepths)
        if self.depth_range[0] < self.depth_range[1]:
            depth_min = self.depth_range[0]
            depth_max = self.depth_range[1]

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] = stage3_rpc[:, 0] / 4
        stage3_rpc[:, 1] = stage3_rpc[:, 1] / 4
        stage3_rpc[:, 5] = stage3_rpc[:, 5] / 4
        stage3_rpc[:, 6] = stage3_rpc[:, 6] / 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outrpc,
                "out_view": out_view,
                "out_name": out_name
                }

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            return self.get_sample(idx)
        else:
            return self.get_pred_sample(idx)
