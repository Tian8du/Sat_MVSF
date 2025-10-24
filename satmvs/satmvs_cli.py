# -*- coding: utf-8 -*-
"""
CLI entry for Sat-MVSF.

Key features:
- Windows-safe GDAL/PROJ bootstrap (no-op on Linux/macOS).
- Resource path resolver compatible with PyInstaller and source runs.
- Pure-function runner `run_satmvsf(...)` for programmatic calls.
- Robust argument parsing (boolean flags via store_true/false).
- Friendly errors for missing inputs and empty scenes.
- Optional PyTorch import (works in pure-CPU environments).
- Per-pair timing + aggregated JSON result for scripting.
"""

from __future__ import annotations

import os
import sys
import glob
import gc
import time
import json
import ctypes
import argparse
import traceback
import platform
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# ----------------- Optional heavy deps (PyTorch) -----------------
try:
    import torch  # type: ignore
except Exception:
    torch = None  # Allow pure-CPU environments without torch installed


# ----------------- Windows-only GDAL bootstrap -----------------
def _bootstrap_gdal() -> None:
    """
    On Windows + (conda/miniconda), pre-load GDAL/GEOS/PROJ DLLs and set
    GDAL_DATA/PROJ_LIB. No-op on Linux/macOS.
    """
    if platform.system() != "Windows":
        return

    prefix = sys.prefix
    libbin = os.path.join(prefix, "Library", "bin")
    if not os.path.isdir(libbin):
        # Not a conda-style layout; skip
        return

    # Add DLL search path (Python 3.8+)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(libbin)  # type: ignore[attr-defined]
    else:
        os.environ["PATH"] = libbin + os.pathsep + os.environ.get("PATH", "")

    # Provide default GDAL/PROJ data directories if not set
    os.environ.setdefault("GDAL_DATA", os.path.join(prefix, "Library", "share", "gdal"))
    os.environ.setdefault("PROJ_LIB", os.path.join(prefix, "Library", "share", "proj"))

    def _load(pattern: str) -> None:
        matches = glob.glob(os.path.join(libbin, pattern))
        if matches:
            ctypes.CDLL(matches[0])

    # Attempt to pre-load common dependency DLLs (best-effort)
    for pat in ("gdal*.dll", "geos_c*.dll", "proj*.dll", "hdf5*.dll", "libcurl*.dll", "zlib*.dll", "iconv*.dll"):
        try:
            _load(pat)
        except OSError as e:
            print(">>> GDAL DLL dependency problem:", e, file=sys.stderr)
            raise


# ----------------- Resource path helper -----------------
def resource_path(rel_path: str) -> str:
    """
    Resolve paths for both source runs and PyInstaller bundles.
    If `rel_path` is absolute, return it as-is; otherwise resolve relative to this file (or _MEIPASS).
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    p = Path(rel_path)
    return str((p if p.is_absolute() else (base / rel_path)).resolve())


# ----------------- Project imports (internal) -----------------
from satmvs.utils.files import get_all_files, ensure_forward_slash, mkdir_if_not_exist
from satmvs.pipeline.rpc_pipeline import Pipeline
from satmvs.utils.io import (
    read_info_from_txt, read_pair_from_txt, read_border_from_txt,
    read_range_from_txt, read_config
)
from satmvs.pylog.logger import Logger


# ----------------- Helpers -----------------
def _build_pairs(
    image_info_file: str,
    camera_info_file: str,
    pair_info_file: str
) -> Tuple[List[List[str]], List[List[str]], List[List[int]]]:
    """
    Build (image_pairs, camera_pairs, id_pairs) from info files.
    """
    image_paths = read_info_from_txt(image_info_file)
    camera_paths = read_info_from_txt(camera_info_file)
    pair_info = read_pair_from_txt(pair_info_file)

    image_pairs = [[image_paths[int(i)] for i in pair_info[p]] for p in range(len(pair_info))]
    camera_pairs = [[camera_paths[int(i)] for i in pair_info[p]] for p in range(len(pair_info))]
    id_pairs = [[int(i) for i in pair_info[p]] for p in range(len(pair_info))]
    return image_pairs, camera_pairs, id_pairs


# ----------------- Core runner (pure function) -----------------
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
) -> Dict[str, Any]:
    """
    Programmatic entry point.
    - Does not read global argparse state.
    - Raises exceptions on failure; returns a dict on success.
    - Suitable for Python or C++ subprocess orchestration.

    Returns:
        {
          "ok": True,
          "workspace": "<abs path>",
          "records": [
              {"scene": str, "pair": str, "output": str, "log": str, "elapsed_sec": float},
              ...
          ]
        }
    """
    _bootstrap_gdal()

    # Resolve resource-dependent files (support PyInstaller)
    config_file = resource_path(config_file)
    loadckpt = resource_path(loadckpt)

    # Normalize/prepare directories
    info_root = ensure_forward_slash(info_root)
    workspace = ensure_forward_slash(workspace)
    mkdir_if_not_exist(workspace)

    # Validate existence
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not os.path.isdir(info_root):
        raise FileNotFoundError(f"Info root directory not found: {info_root}")

    # Optional device handling
    if device:
        # If CUDA requested while torch or CUDA is unavailable, fail early
        if device.startswith("cuda"):
            if (torch is None) or (not torch.cuda.is_available()):
                raise RuntimeError("CUDA requested but PyTorch CUDA is not available.")
            # Expose CUDA device index to internal modules if needed
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else device

    # Load config
    config = read_config(config_file)

    # Enumerate scenes (directories only)
    scene_names = [d for d in os.listdir(info_root) if os.path.isdir(os.path.join(info_root, d))]
    if not scene_names:
        raise FileNotFoundError(f"No scene directories found under: {info_root}")

    run_records: List[Dict[str, Any]] = []

    for scene in scene_names:
        scene_root = f"{info_root}/{scene}"

        # Project metadata (.prj); enforce existence
        prjs = get_all_files(scene_root, ".prj")
        if not prjs:
            raise FileNotFoundError(f"No .prj found in scene: {scene_root}")
        prj_file = prjs[0]
        with open(prj_file, "r", encoding="utf-8", errors="ignore") as f:
            prj_str = f.read()

        # Required info files
        images_info_file = f"{scene_root}/images_info.txt"
        cameras_info_file = f"{scene_root}/cameras_info.txt"
        pairs_info_file = f"{scene_root}/pair.txt"
        border_info_file = f"{scene_root}/border.txt"
        range_file = f"{scene_root}/range.txt"

        for fpath in (images_info_file, cameras_info_file, pairs_info_file, border_info_file, range_file):
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Missing required info file: {fpath}")

        image_pair_list, camera_pair_list, id_pair_list = _build_pairs(
            images_info_file, cameras_info_file, pairs_info_file
        )
        border_info = read_border_from_txt(border_info_file)
        depth_range = read_range_from_txt(range_file)

        pair_workspace = f"{workspace}/{scene}"
        mkdir_if_not_exist(pair_workspace)

        # Iterate over image/camera pairs
        for image_paths, camera_paths, idxs in zip(image_pair_list, camera_pair_list, id_pair_list):
            out_name = "_".join(map(str, idxs))
            output = f"{pair_workspace}/{out_name}"
            mkdir_if_not_exist(output)

            # Logger for each pair
            log_file = ensure_forward_slash(os.path.join(output, "workspace_log.txt"))
            logger = Logger(log_file)

            # Log key context for reproducibility
            logger.info(f"config: {config}")
            for p_img, p_cam in zip(image_paths, camera_paths):
                logger.info(f"  {p_img}  {p_cam}")
            logger.info(f"output: {output}")
            logger.info(
                f"border: start ({border_info[0]}, {border_info[1]}) "
                f"xsize {border_info[2]} ysize {border_info[3]} "
                f"xuint {border_info[4]} yuint {border_info[5]}"
            )
            logger.info(
                f"search range: [{depth_range[0]}]-[{depth_range[1]}] interval:{depth_range[2]}"
            )

            # Emulate argparse namespace for Pipeline if needed
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

            # Run pipeline with timing
            t_pair = time.time()
            pipeline = Pipeline(
                image_paths, camera_paths, config, prj_str,
                border_info, depth_range, output, logger, _a
            )
            pipeline.run()
            elapsed = round(time.time() - t_pair, 3)

            run_records.append({
                "scene": scene,
                "pair": out_name,
                "output": ensure_forward_slash(output),
                "log": log_file,
                "elapsed_sec": elapsed
            })

            # Cleanup
            del pipeline
            gc.collect()
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {"ok": True, "workspace": ensure_forward_slash(workspace), "records": run_records}


# ----------------- CLI entry -----------------
def main() -> None:
    p = argparse.ArgumentParser("Sat-MVSF")
    # Required (with defaults)
    p.add_argument("--config", default="config/config.json", help="Path to configuration file")
    p.add_argument("--info_root", default="infos/ZY3-WHU", help="Root folder containing scene infos")
    p.add_argument("--workspace", default="temp_workspace_WHUZY3", help="Output workspace directory")
    p.add_argument("--checkpoint", default="checkpoints/casred.ckpt", help="Path to model checkpoint")
    p.add_argument("--device", default=None, help="Device spec, e.g., 'cuda:0' or 'cpu' (None = leave as-is)")

    # Optional numeric hyper-parameters
    p.add_argument("--resize_scale", type=float, default=1.0, help="Output scale for depth/image (W,H)")
    p.add_argument("--sample_scale", type=float, default=1.0, help="Downsample scale for cost volume (W,H)")
    p.add_argument("--interval_scale", type=float, default=2.5, help="Depth interval scale")
    p.add_argument("--batch_size", type=int, default=1, help="Prediction batch size")

    # Optional booleans via store_{true,false}
    p.add_argument("--share_cr", action="store_true", help="Whether to share the cost-volume regularization")
    p.add_argument("--adaptive_scaling", dest="adaptive_scaling", action="store_true",
                   help="Adapt input size to the network (enable)")
    p.add_argument("--no-adaptive_scaling", dest="adaptive_scaling", action="store_false",
                   help="Disable adaptive input scaling")
    p.set_defaults(adaptive_scaling=True)

    # Optional strings (lists encoded as 'a,b,c')
    p.add_argument("--ndepths", default="64,32,8", help="Number of depth hypotheses per stage, e.g., '64,32,8'")
    p.add_argument("--depth_inter_r", default="4,2,1", help="Depth interval ratios per stage, e.g., '4,2,1'")
    p.add_argument("--cr_base_chs", default="8,8,8", help="Cost-regularization base channels, e.g., '8,8,8'")

    args = p.parse_args()

    t0 = time.time()
    try:
        result = run_satmvsf(
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
        result["elapsed_sec"] = round(time.time() - t0, 3)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
