import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # 防止无显示环境报错
from collections import OrderedDict
from tqdm import tqdm

from satmvs.networks.casred import Infer_CascadeREDNet
from satmvs.dataset.rpc_dataset import MVSDataset
from satmvs.utils.utils import *           # 如果里面有 tocuda(…) 会用到，但我们重写一个本地版本以避免强制 .cuda()
from satmvs.utils.io import save_pfm

# ------------------- 设备选择 -------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==> Using device:", DEVICE)
cudnn.benchmark = (DEVICE.type == "cuda")

# 递归把 Python 容器中的 tensor / ndarray 移到 device（不依赖 utils.tocuda）
def to_device(obj, device):
    import torch
    import numpy as np
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device(v, device) for v in obj]
        return type(obj)(t)  # 保持原容器类型
    return obj

def load_ckpt_safely(model, ckpt_path, logger, strict=False, device=DEVICE):
    # 1) 优先安全模式（老版 torch 无 weights_only）
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    # 2) 取 state_dict
    sd = ckpt.get("model", ckpt)

    # 3) 去 DataParallel 前缀
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing:
        logger.info(f"[loadckpt] missing keys: {missing}")
    if unexpected:
        logger.info(f"[loadckpt] unexpected keys: {unexpected}")

@make_nograd_func
def test_sample(model, sample, device=DEVICE):
    model.eval()
    sample_dev = to_device(sample, device)
    outputs = model(sample_dev["imgs"], sample_dev["cam_para"], sample_dev["depth_values"])
    depth_est = outputs["stage3"]["depth"]
    photometric_confidence = outputs["stage3"]["photometric_confidence"]

    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "ref_img": sample_dev["imgs"][:, 0],
    }
    saved_outputs = {
        "outimage": sample["outimage"],   # 这些通常是路径/标量，保留原对象
        "outcam": sample["outcam"],
    }
    return image_outputs, saved_outputs

def test(testpath, depth_range, view_num, args, logger):
    logger.info("argv:{}".format(sys.argv[1:]))

    # dataset & dataloader
    pre_dataset = MVSDataset(testpath, "pred", view_num, depth_range, args)
    # CPU/GPU 都没问题；如果以后在 GPU 上跑，可把 num_workers 调高
    Pre_ImgLoader = DataLoader(
        pre_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=(DEVICE.type == "cuda")
    )

    logger.info("===============> Model: Cascade RED Net ===========>")
    model = Infer_CascadeREDNet(
        min_interval=depth_range[2],
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
        share_cr=args.share_cr,
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
    )

    # 只在有 CUDA 且多卡时用 DataParallel
    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
    else:
        model = model.to(DEVICE)

    logger.info(f"loading model {args.loadckpt}")
    load_ckpt_safely(model, args.loadckpt, logger, strict=False, device=DEVICE)
    logger.info('Number of model parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    # output folder
    output_folder = os.path.join(testpath, 'mvs')
    os.makedirs(output_folder, exist_ok=True)

    with tqdm(total=len(pre_dataset), desc="MVS") as pbar:
        for batch_idx, sample in enumerate(Pre_ImgLoader):
            bview = sample['out_view'][0]
            bname = sample['out_name'][0]

            image_outputs, saved_outputs = test_sample(model, sample, device=DEVICE)

            # 保存结果
            depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
            prob      = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

            out2 = os.path.join(output_folder, bview)
            os.makedirs(os.path.join(out2, 'prob', 'color'), exist_ok=True)
            os.makedirs(os.path.join(out2, 'init', 'color'), exist_ok=True)
            os.makedirs(os.path.join(out2, 'depth_masked'), exist_ok=True)

            init_depth_map_path = os.path.join(out2, 'init', f'{bname}.pfm')
            prob_map_path       = os.path.join(out2, 'prob', f'{bname}.pfm')

            save_pfm(init_depth_map_path, depth_est)
            save_pfm(prob_map_path, prob)

            # plt.imsave(os.path.join(out2, 'init', 'color', f'{bname}.png'), depth_est, format='png')
            # plt.imsave(os.path.join(out2, 'prob', 'color', f'{bname}_prob.png'), prob, format='png')

            del image_outputs, saved_outputs
            pbar.update(args.batch_size)

if __name__ == '__main__':
    pass
