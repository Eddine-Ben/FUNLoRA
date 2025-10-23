import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import random
import logging
import ctypes
import platform
from typing import Optional

import requests
from tqdm import tqdm

import numpy as np
import torch as th


def create_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    return logger


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = False
    th.backends.cudnn.benchmark = True
    try:
        th.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def setup_cuda_optimizations():
    if th.cuda.is_available():
        th.backends.cudnn.benchmark = True
        th.backends.cudnn.deterministic = False
        th.cuda.empty_cache()
        try:
            print(f"🚀 CUDA Device: {th.cuda.get_device_name()} • "
                  f"VRAM Alloc {th.cuda.memory_allocated()/1024**3:.2f} GB")
        except Exception:
            print("🚀 CUDA ready")


def cleanup_memory():
    # cv2 threads off handled in dataloader worker init
    for _ in range(3):
        if th.cuda.is_available():
            th.cuda.empty_cache()
        import gc as _gc
        _gc.collect()
    try:
        if os.name == "nt":
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)  # type: ignore[attr-defined]
    except Exception:
        pass


def monitor_system_resources() -> str:
    if th.cuda.is_available():
        used = th.cuda.memory_allocated()
        total = th.cuda.mem_get_info()[1]
        r = (used / total * 100) if total > 0 else 0
        return f"🖥️ GPU {r:.1f}% • {used/1e9:.2f}/{total/1e9:.2f} GB"
    return "🖥️ GPU N/A"


def download_file(url: str, save_path: str, timeout: int = 90):
    with requests.Session() as session:
        r = session.get(url, stream=True, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f, tqdm(
            desc=f"⬇️ {os.path.basename(save_path)}", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    pbar.update(f.write(chunk))
    print(f"✅ Downloaded: {save_path}")


def can_use_torch_compile() -> bool:
    if platform.system().lower().startswith("win"):
        return False
    return getattr(th.version, "triton", None) is not None

