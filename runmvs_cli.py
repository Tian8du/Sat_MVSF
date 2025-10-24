import os, sys, glob, ctypes, pathlib
from pathlib import Path

def _bootstrap_gdal():
    PREFIX = sys.prefix
    LIBBIN = os.path.join(PREFIX, "Library", "bin")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(LIBBIN)
    else:
        os.environ["PATH"] = LIBBIN + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("GDAL_DATA", os.path.join(PREFIX, "Library", "share", "gdal"))
    os.environ.setdefault("PROJ_LIB",  os.path.join(PREFIX, "Library", "share", "proj"))
    def _load(pattern):
        m = glob.glob(os.path.join(LIBBIN, pattern))
        if not m: return
        ctypes.CDLL(m[0])
    for pat in ("gdal*.dll", "geos_c*.dll", "proj*.dll", "hdf5*.dll", "libcurl*.dll", "zlib*.dll", "iconv*.dll"):
        try: _load(pat)
        except OSError as e:
            print(">>> GDAL DLL dependency problem:", e, file=sys.stderr); raise

# ----------------- 通用工具 -----------------
def resource_path(rel_path: str) -> str:
    """兼容 PyInstaller / 源码运行"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    p = Path(rel_path)
    return str(p if p.is_absolute() else (base / rel_path))

# ----------------- 核心依赖 -----------------
from utils.files import get_all_files, ensure_forward_slash, mkdir_if_not_exist
from pipeline.rpc_pipeline import Pipeline
from utils.io import (
    read_info_from_txt, read_pair_from_txt, read_border_from_txt,
    read_range_from_txt, read_config
)
from pylog.logger import Logger
import gc, torch, argparse, json, time, traceback

# ----------------- 业务子函数 -----------------
def _sparse_pair(image_info_file, camera_info_file, pair_info_file):
    image_paths  = read_info_from_txt(image_info_file)
    camera_paths = read_info_from_txt(camera_info_file)
    pair_info    = read_pair_from_txt(pair_info_file)
    image_pairs  = [[image_paths[int(i)]  for i in pair_info[p]] for p in range(len(pair_info))]
    camera_pairs = [[camera_paths[int(i)] for i in pair_info[p]] for p in range(len(pair_info))]
    id_pairs     = [[int(i) for i in pair_info[p]]            for p in range(len(pair_info))]
    return image_pairs, camera_pairs, id_pairs
from typing import Optional

def run_satmvsf(
    *,
    config_file: str,
    info_root: str,
    workspace: str,
    loadckpt: str = "checkpoints/casred.ckpt",
    device: Optional[str] = None,
    resize_scale: float = 1.0,
    sample_scale: float = 1.0,
    interval_scale: float = 2.5,
    batch_size: int = 1,
    adaptive_scaling: bool = True,
    share_cr: bool = False,
    ndepths: str = "64,32,8",
    depth_inter_r: str = "4,2,1",
    cr_base_chs: str = "8,8,8",
):
    """
    供 Python/C++ 上层调用的**纯函数式**入口：
    - 不读全局 args，不写死路径，不 sys.exit
    - 失败抛异常，成功返回一个结果字典（便于 JSON 序列化）
    """
    _bootstrap_gdal()

    # 资源路径适配
    config_file = resource_path(config_file)
    loadckpt    = resource_path(loadckpt)

    # 统一 workspace / 目录
    info_root = ensure_forward_slash(info_root)
    workspace = ensure_forward_slash(workspace)
    mkdir_if_not_exist(workspace)

    # 可选：GPU 选择（让上层传入；不强行写死）
    if device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but device requested: " + device)
        # 供内部模块读取
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else device

    # 读配置
    config = read_config(config_file)

    # 遍历场景
    scene_names = os.listdir(info_root)
    run_records = []  # 收集每个pair的输出目录/日志等
    for scene in scene_names:
        scene_root = f"{info_root}/{scene}"
        prj_file = get_all_files(scene_root, ".prj")[0]
        with open(prj_file, "r") as f:
            prj_str = f.read()

        images_info_file  = f"{scene_root}/images_info.txt"
        cameras_info_file = f"{scene_root}/cameras_info.txt"
        pairs_info_file   = f"{scene_root}/pair.txt"
        border_info_file  = f"{scene_root}/border.txt"
        range_file        = f"{scene_root}/range.txt"

        image_pair_list, camera_pair_list, id_pair_list = _sparse_pair(
            images_info_file, cameras_info_file, pairs_info_file
        )
        border_info = read_border_from_txt(border_info_file)
        depth_range = read_range_from_txt(range_file)

        pair_workspace = f"{workspace}/{scene}"
        mkdir_if_not_exist(pair_workspace)

        for image_paths, camera_paths, idxs in zip(image_pair_list, camera_pair_list, id_pair_list):
            out_name = "_".join(map(str, idxs))
            output   = f"{pair_workspace}/{out_name}"
            mkdir_if_not_exist(output)

            logger = Logger(ensure_forward_slash(os.path.join(output, "workspace_log.txt")))
            # 记录关键信息
            logger.info(f"config: {config}")
            for p1, p2 in zip(image_paths, camera_paths):
                logger.info(f"  {p1}  {p2}")
            logger.info(f"output: {output}")
            logger.info(
                f"border: start ({border_info[0]}, {border_info[1]}) "
                f"xsize {border_info[2]} ysize {border_info[3]} "
                f"xuint {border_info[4]} yuint {border_info[5]}"
            )
            logger.info(
                f"search range: [{depth_range[0]}]-[{depth_range[1]}] interval:{depth_range[2]}"
            )

            # 将原先依赖 argparse 的参数以对象形式传入（如你的 Pipeline 需要）
            class _Args:
                pass
            _a = _Args()
            _a.resize_scale = resize_scale
            _a.sample_scale = sample_scale
            _a.interval_scale = interval_scale
            _a.batch_size = batch_size
            _a.adaptive_scaling = adaptive_scaling
            _a.share_cr = share_cr
            _a.ndepths = ndepths
            _a.depth_inter_r = depth_inter_r
            _a.cr_base_chs = cr_base_chs
            _a.loadckpt = loadckpt
            _a.config_file = config_file
            _a.info_root = info_root
            _a.workspace = workspace

            pipeline = Pipeline(
                image_paths, camera_paths, config, prj_str,
                border_info, depth_range, output, logger, _a
            )
            pipeline.run()

            run_records.append({"scene": scene, "pair": out_name, "output": output})

            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {"ok": True, "workspace": workspace, "records": run_records}

# ----------------- 命令行入口（供人/脚本/C++子进程用） -----------------
def main():
    p = argparse.ArgumentParser("Sat-MVSF")

    # === 必需参数（都有默认） ===
    p.add_argument("--config", default="config/config.json", help="Path to configuration file")
    p.add_argument("--info_root", default="infos/ZY3-WHU", help="Root folder containing scene infos")
    p.add_argument("--workspace", default="temp_workspace_WHUZY3", help="Output workspace directory")
    p.add_argument("--checkpoint", default="checkpoints/casred.ckpt", help="Path to model checkpoint")
    p.add_argument("--device", default="cuda:0", help="Device: e.g., cuda:0 or cpu")

    # === 可选参数（与原逻辑保持一致） ===
    p.add_argument('--resize_scale', type=float, default=1.0, help='Output scale for depth and image (W,H)')
    p.add_argument('--sample_scale', type=float, default=1.0, help='Downsample scale for cost volume (W,H)')
    p.add_argument('--interval_scale', type=float, default=2.5, help='Depth interval scale')
    p.add_argument('--batch_size', type=int, default=1, help='Predict batch size')
    p.add_argument('--adaptive_scaling', type=bool, default=True, help='Adapt image size to network')
    p.add_argument('--share_cr', action='store_true', help='Whether share the cost volume regularization')
    p.add_argument('--ndepths', default="64,32,8", help='Number of depth hypotheses per stage')
    p.add_argument('--depth_inter_r', default="4,2,1", help='Depth interval ratios')
    p.add_argument('--cr_base_chs', default="8,8,8", help='Cost regularization base channels')

    args = p.parse_args()

    t0 = time.time()
    try:
        ret = run_satmvsf(
            config_file=args.config,
            info_root=args.info_root,
            workspace=args.workspace,
            loadckpt=args.checkpoint,
            device=args.device,
            resize_scale=args.resize_scale,
            sample_scale=args.sample_scale,
            interval_scale=args.interval_scale,
            batch_size=args.batch_size,
            adaptive_scaling=args.adaptive_scaling,
            share_cr=args.share_cr,
            ndepths=args.ndepths,
            depth_inter_r=args.depth_inter_r,
            cr_base_chs=args.cr_base_chs,
        )
        ret["elapsed_sec"] = round(time.time()-t0, 3)
        print(json.dumps(ret), flush=True);  sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
