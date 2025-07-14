import os
import argparse
from utils.files import get_all_files, ensure_forward_slash, mkdir_if_not_exist
from pipeline.rpc_pipeline import Pipeline
from utils.io import read_info_from_txt, read_pair_from_txt, read_border_from_txt, read_range_from_txt
from utils.io import read_config
from pylog.logger import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='Sat-MVSF')
parser.add_argument("--config_file", default="config/config.json")
# infomation for input data
parser.add_argument("--info_root", default="infos/ISPRS_ZY3_test")

# model
parser.add_argument('--loadckpt', default="checkpoints/casred.ckpt",
                    help='load a specific checkpoint')
# load data parameters
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='predict batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True,
                    help='Let image size to fit the network, including scaling and cropping')
# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
# output
parser.add_argument("--workspace", type=str, default=r"E:\Codes2\sat-mvsf\temp_workspace_test")

# parse arguments and check
args = parser.parse_args()


def sparse_pair(image_info_file, camera_info_file, pair_info_file):
    image_paths = read_info_from_txt(image_info_file)
    camera_paths = read_info_from_txt(camera_info_file)
    pair_info = read_pair_from_txt(pair_info_file)

    image_pairs = [[image_paths[int(img_id)] for img_id in pair_info[pair_id]] for pair_id in
                   range(len(pair_info))]
    camera_pairs = [[camera_paths[int(cam_id)] for cam_id in pair_info[pair_id]] for pair_id in
                    range(len(pair_info))]
    id_pairs = [[int(idx) for idx in pair_info[pair_id]] for pair_id in range(len(pair_info))]

    image_pairs = [[image_paths[int(img_id)].replace("D:", "F:") for img_id in pair_info[pair_id]] for
                   pair_id in range(len(pair_info))]
    camera_pairs = [[camera_paths[int(cam_id)].replace("D:", "F:") for cam_id in pair_info[pair_id]]
                    for pair_id in range(len(pair_info))]
    id_pairs = [[int(idx) for idx in pair_info[pair_id]] for pair_id in range(len(pair_info))]

    return image_pairs, camera_pairs, id_pairs


if __name__ == "__main__":
    # read config
    info_root = ensure_forward_slash(args.info_root)
    workspace = ensure_forward_slash(args.workspace)
    
    config = read_config(args.config_file)
    
    mkdir_if_not_exist(args.workspace)
    
    scene_names = os.listdir(args.info_root)
    for scene in scene_names:
        scene_root = "{}/{}".format(args.info_root, scene)
        prj_file  = get_all_files(scene_root, ".prj")[0]
        
        # read wtk projection info
        with open(prj_file, "r") as f:
            prj_str = f.read()
        
        images_info_file = "{}/{}".format(scene_root, "images_info.txt")
        cameras_info_file = "{}/{}".format(scene_root, "cameras_info.txt")
        pairs_info_file = "{}/{}".format(scene_root, "pair.txt")
        border_info_file = "{}/{}".format(scene_root, "border.txt")
        range_file = "{}/{}".format(scene_root, "range.txt")
        
        cameras_info_file = ensure_forward_slash(os.path.join(scene_root, "cameras_info.txt"))
        pairs_info_file = ensure_forward_slash(os.path.join(scene_root, "pair.txt"))
        border_info_file = ensure_forward_slash(os.path.join(scene_root, "border.txt"))
        range_file = ensure_forward_slash(os.path.join(scene_root, "range.txt"))
        
        image_pair_list, camera_pair_list, id_pair_list = sparse_pair(
            images_info_file, cameras_info_file, pairs_info_file)
        
        border_info = read_border_from_txt(border_info_file)
        depth_range = read_range_from_txt(range_file)
        
        pair_workspace = "{}/{}".format(workspace, scene)
        mkdir_if_not_exist(pair_workspace)
        
        out_name = ""
        for image_paths, camera_paths, idxs in zip(image_pair_list, camera_pair_list, id_pair_list):
            for idx in idxs:
                out_name += str(idx)+"_"
            
            out_name = out_name[:-1]
            output = "{}/{}".format(pair_workspace, out_name)
            mkdir_if_not_exist(output)
                
            logger = Logger(ensure_forward_slash(os.path.join(output, "workspace_log.txt")))
                
            # output some information
            logger.info("config: {}".format(config))
            logger.info("Data paths:")
            for p1, p2 in zip(image_paths, camera_paths):
                logger.info("  {}  {}".format(p1, p2))
                    
            logger.info("output: {}".format(output))
            logger.info("border: start at ({}, {}) xsize {}  ysize {} xuint {}  yuint {}".format(
                border_info[0], border_info[1], border_info[2], border_info[3], border_info[4], border_info[5]))
            logger.info("search range: [{}]-[{}] interval:{}".format(depth_range[0], depth_range[1], depth_range[2]))
            pipeline = Pipeline(image_paths, camera_paths, config, prj_str,
                                border_info, depth_range, output, logger, args)
            pipeline.run()