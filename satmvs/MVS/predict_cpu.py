"""
This is a script for running the Sat-MVSF.
Copyright (C) <2023> <Jian Gao & GPCV>
... (license 省略)
"""

import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn

from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from satmvs.networks.casred import Infer_CascadeREDNet

from satmvs.utils.utils import *
from satmvs.utils.io import save_pfm
from satmvs.dataset.rpc_dataset import MVSDataset

# ----------------------------
# 🔧 新增：通用的 to_device 搬运函数（替代 tocuda）
# ----------------------------
def to_device(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=False)
    if isinstance(sample, dict):
        return {k: to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, (list, tuple)):
        return type(sample)(to_device(v, device) for v in sample)
    return sample

# 🔧：在 CPU 场景不使用 cudnn
cudnn.benchmark = torch.cuda.is_available()

def load_ckpt_safely(model, ckpt_path, logger, device, strict=False):
    # 1) 兼容新老 torch.load
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    # 2) 取出 state_dict
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # 3) 去掉 DataParallel 的 'module.' 前缀（若有）
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing:
        logger.info(f"[loadckpt] missing keys: {missing}")
    if unexpected:
        logger.info(f"[loadckpt] unexpected keys: {unexpected}")

def test(testpath, depth_range, view_num, args, logger):
    logger.info("argv:{}".format(sys.argv[1:]))

    # ----------------------------
    # 🔧 新增：device 选择 + 参数微调（CPU 建议 batch_size=1, num_workers=0）
    # ----------------------------
    use_gpu_flag = getattr(args, "use_gpu", True)  # 允许你通过 --use_gpu 关闭 GPU
    device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu_flag) else "cpu")
    if device.type == "cpu":
        logger.info("Running on CPU. Expect slower inference; consider args.batch_size=1, num_workers=0.")

    # dataset, dataloader
    pre_dataset = MVSDataset(testpath, "pred", view_num, depth_range, args)
    num_workers = 0 if device.type == "cpu" else 0  # Windows/CPU 更稳妥用 0
    Pre_ImgLoader = DataLoader(pre_dataset, args.batch_size, shuffle=False,
                               num_workers=num_workers, drop_last=False)

    # build model
    model = Infer_CascadeREDNet(min_interval=depth_range[2],
                                ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                                depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                                cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    logger.info("===============> Model: Cascade RED Net ===========>")

    # ----------------------------
    # 🔧 修改：仅在多卡 + CUDA 时使用 DataParallel
    # ----------------------------
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # ----------------------------
    # 🔧 修改：健壮地加载 checkpoint（map_location=device）
    # ----------------------------
    logger.info(f"loading model {args.loadckpt}")
    try:
        state_dict = torch.load(args.loadckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.loadckpt, map_location=device)

    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    try:
        model.load_state_dict(state_dict)
        print(" Loaded model without 'module.' adjustment.")
    except RuntimeError:
        print(" Failed to load directly, trying to remove 'module.' prefix...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v if k.startswith('module.') else v
        model.load_state_dict(new_state_dict)
        print(" Loaded model after adjusting 'module.' prefix.")

    param_cnt = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {param_cnt}')
    logger.info(f'Number of model parameters: {param_cnt}')

    # create output folder
    output_folder = os.path.join(testpath, 'mvs')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    with tqdm(total=len(pre_dataset)) as pbar:
        pbar.set_description("MVS: ")
        for batch_idx, sample in enumerate(Pre_ImgLoader):
            bview = sample['out_view'][0]
            bname = sample['out_name'][0]

            # 推理
            image_outputs, saved_outputs = test_sample(model, sample, device)

            # 保存结果
            depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
            prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

            # paths
            output_folder2 = os.path.join(output_folder, f'{bview}')
            os.makedirs(os.path.join(output_folder2, 'prob', 'color'), exist_ok=True)
            os.makedirs(os.path.join(output_folder2, 'init', 'color'), exist_ok=True)
            os.makedirs(os.path.join(output_folder2, 'depth_masked'), exist_ok=True)

            init_depth_map_path = os.path.join(output_folder2, 'init', f'{bname}.pfm')
            prob_map_path = os.path.join(output_folder2, 'prob', f'{bname}.pfm')

            save_pfm(init_depth_map_path, depth_est)
            save_pfm(prob_map_path, prob)

            del image_outputs, saved_outputs
            pbar.update(args.batch_size)

# 🔧 修改：接收 device，使用 to_device 而不是 tocuda
@make_nograd_func
def test_sample(model, sample, device):
    model.eval()
    sample_dev = to_device(sample, device)
    outputs = model(sample_dev["imgs"], sample_dev["cam_para"], sample_dev["depth_values"])
    depth_est = outputs["stage3"]["depth"]
    photometric_confidence = outputs["stage3"]["photometric_confidence"]
    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "ref_img": sample_dev["imgs"][:, 0]
    }
    saved_outputs = {"outimage": sample["outimage"], "outcam": sample["outcam"]}
    return image_outputs, saved_outputs

if __name__ == '__main__':
    pass
