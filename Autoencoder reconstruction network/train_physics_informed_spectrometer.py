#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a robust physics-informed decoder for a speckle spectrometer.

Forward model
-------------
    I = T S + eta
    z = C I = (C T) S + C eta

During training, the fixed encoder uses W_eff = C @ T.  Only the decoder is
optimized.  The projection matrix C, the effective matrix W_eff, the exact
wavelength axis, and all run settings are saved so that a later inference
program can execute
    experimental speckle I -> C I -> decoder -> reconstructed spectrum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FORMAT_VERSION = 3

# One-click defaults resolved from the script location, not the IDE's working
# directory. Place this file directly under ``experiment3``.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_T_PATH = SCRIPT_DIR / "neural_network" / "1520-1580" / "T_ori.npy"
DEFAULT_OLD_RUN_DIR = SCRIPT_DIR / "runs" / "metafiber_1520-1580"
DEFAULT_ROBUST_RUN_DIR = SCRIPT_DIR / "runs" / "metafiber_1520-1580_robust"
DEFAULT_PROJECTION_PATH = DEFAULT_OLD_RUN_DIR / "physics" / "projection_matrix.npy"


# -----------------------------------------------------------------------------
# Reproducibility and I/O
# -----------------------------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        # TF32 materially accelerates the one-time C @ T construction.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)
    os.replace(tmp, path)


def atomic_torch_save(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def short_file_fingerprint(path: Path) -> Dict:
    """Fast identity record; avoids hashing a transmission matrix of tens of GB."""
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "stat_sha256": hashlib.sha256(payload).hexdigest(),
    }


def environment_info() -> Dict:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
    }


# -----------------------------------------------------------------------------
# Transmission-matrix access and fixed projection construction
# -----------------------------------------------------------------------------


class TransmissionMatrix:
    """Memory-conscious view that always exposes T as [pixels, wavelengths]."""

    def __init__(
        self,
        path: str,
        layout: str = "auto",
        npz_key: Optional[str] = None,
        expected_wavelengths: Optional[int] = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Transmission matrix not found: {self.path}")

        suffix = self.path.suffix.lower()
        self._npz_handle = None
        if suffix == ".npy":
            self.array = np.load(self.path, mmap_mode="r")
        elif suffix == ".npz":
            self._npz_handle = np.load(self.path)
            keys = list(self._npz_handle.keys())
            if not keys:
                raise ValueError(f"No arrays found in {self.path}")
            key = npz_key or keys[0]
            if key not in self._npz_handle:
                raise KeyError(f"npz key {key!r} not found. Available keys: {keys}")
            print("Warning: .npz cannot be memory-mapped; the full T may enter RAM.")
            self.array = self._npz_handle[key]
        else:
            raise ValueError("For large matrices, T_path must be .npy or .npz")

        if self.array.ndim not in (2, 3):
            raise ValueError(f"T must be 2-D or 3-D, got shape {self.array.shape}")

        self.layout = self._resolve_layout(layout, expected_wavelengths)
        if self.array.ndim == 2:
            if self.layout == "pixels_by_wavelength":
                self.n_pixels, self.n_wavelengths = self.array.shape
            else:
                self.n_wavelengths, self.n_pixels = self.array.shape
        else:
            if self.layout == "wavelength_last":
                self.n_pixels = int(np.prod(self.array.shape[:-1]))
                self.n_wavelengths = int(self.array.shape[-1])
            else:
                self.n_wavelengths = int(self.array.shape[0])
                self.n_pixels = int(np.prod(self.array.shape[1:]))

    def _resolve_layout(self, layout: str, expected: Optional[int]) -> str:
        valid = {
            "auto",
            "pixels_by_wavelength",
            "wavelength_by_pixels",
            "wavelength_last",
            "wavelength_first",
        }
        if layout not in valid:
            raise ValueError(f"Unknown T layout {layout!r}")

        if layout != "auto":
            if self.array.ndim == 2 and layout not in {
                "pixels_by_wavelength",
                "wavelength_by_pixels",
            }:
                raise ValueError("2-D T requires pixels_by_wavelength or wavelength_by_pixels")
            if self.array.ndim == 3 and layout not in {"wavelength_last", "wavelength_first"}:
                raise ValueError("3-D T requires wavelength_last or wavelength_first")
            return layout

        shape = self.array.shape
        if expected is not None:
            matches = [i for i, size in enumerate(shape) if size == expected]
            if len(matches) == 1:
                axis = matches[0]
                if self.array.ndim == 2:
                    return "pixels_by_wavelength" if axis == 1 else "wavelength_by_pixels"
                if axis == 0:
                    return "wavelength_first"
                if axis == self.array.ndim - 1:
                    return "wavelength_last"
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous wavelength axis in T shape {shape}; specify --T_layout"
                )

        if self.array.ndim == 2:
            # Spectral dimension is normally smaller than the camera-pixel dimension.
            return "pixels_by_wavelength" if shape[1] <= shape[0] else "wavelength_by_pixels"

        largest_axis = int(np.argmax(shape))
        if largest_axis == 0:
            return "wavelength_first"
        if largest_axis == self.array.ndim - 1:
            return "wavelength_last"
        raise ValueError(f"Cannot infer wavelength axis for T shape {shape}; specify --T_layout")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_pixels, self.n_wavelengths

    def wavelength_block(self, start: int, end: int) -> np.ndarray:
        """Return a [pixels, block_wavelengths] view/copy."""
        if not 0 <= start < end <= self.n_wavelengths:
            raise IndexError((start, end, self.n_wavelengths))

        if self.array.ndim == 2:
            if self.layout == "pixels_by_wavelength":
                block = self.array[:, start:end]
            else:
                block = self.array[start:end, :].T
        elif self.layout == "wavelength_last":
            block = self.array[..., start:end].reshape(self.n_pixels, end - start)
        else:
            block = self.array[start:end, ...].reshape(end - start, self.n_pixels).T

        return np.asarray(block)

    def pixel_block(self, start: int, end: int) -> np.ndarray:
        """Return a [block_pixels, wavelengths] view/copy for quick statistics."""
        if not 0 <= start < end <= self.n_pixels:
            raise IndexError((start, end, self.n_pixels))

        if self.array.ndim == 2:
            if self.layout == "pixels_by_wavelength":
                block = self.array[start:end, :]
            else:
                block = self.array[:, start:end].T
        elif self.layout == "wavelength_last":
            block = self.array.reshape(self.n_pixels, self.n_wavelengths)[start:end, :]
        else:
            block = self.array.reshape(self.n_wavelengths, self.n_pixels)[:, start:end].T
        return np.asarray(block)

    def materialize_pixels_by_wavelength(self) -> np.ndarray:
        """Load T once into writable, C-contiguous FP32 RAM as [pixels, wavelengths]."""
        if self.array.ndim == 2:
            source = (
                self.array
                if self.layout == "pixels_by_wavelength"
                else self.array.T
            )
        elif self.layout == "wavelength_last":
            source = self.array.reshape(self.n_pixels, self.n_wavelengths)
        else:
            source = self.array.reshape(self.n_wavelengths, self.n_pixels).T
        return np.array(source, dtype=np.float32, order="C", copy=True)


def print_transmission_matrix_statistics(
    T: TransmissionMatrix,
    rows_per_block: int = 128,
    num_blocks: int = 3,
) -> None:
    """Print representative raw-value statistics without loading the full T."""
    rows_per_block = min(rows_per_block, T.n_pixels)
    max_start = max(0, T.n_pixels - rows_per_block)
    starts = np.linspace(0, max_start, num_blocks, dtype=np.int64)
    samples = [
        np.array(T.pixel_block(int(s), int(s) + rows_per_block), dtype=np.float32, copy=True)
        for s in starts
    ]
    sample = np.concatenate(samples, axis=0).reshape(-1)
    percentiles = np.percentile(sample, [0.1, 1.0, 50.0, 99.0, 99.9])
    print(
        "T raw-value quick statistics "
        f"(sampled {sample.size:,} values from {len(starts)} pixel blocks):"
    )
    print(f"  stored dtype : {T.array.dtype}")
    print(f"  sample min/max: {float(sample.min()):.8g} / {float(sample.max()):.8g}")
    print(f"  sample mean/std: {float(sample.mean()):.8g} / {float(sample.std()):.8g}")
    print(
        "  percentiles  : "
        f"p0.1={percentiles[0]:.8g}, p1={percentiles[1]:.8g}, "
        f"p50={percentiles[2]:.8g}, p99={percentiles[3]:.8g}, "
        f"p99.9={percentiles[4]:.8g}"
    )
    if sample.min() >= 0 and sample.max() <= 1.5:
        print("  scale hint   : values appear compatible with an approximately [0, 1] scale")
    elif np.issubdtype(T.array.dtype, np.integer) or sample.max() > 2.0:
        print(
            "  scale hint   : values are not on a [0, 1] scale; pixel-noise std must use "
            "these same camera units (or apply one global scale to both T and measurements)"
        )
    else:
        print("  scale hint   : custom/dark-subtracted scale; inspect calibration preprocessing")


def expected_grid_size(start_nm: float, end_nm: float, step_nm: Optional[float]) -> Optional[int]:
    if step_nm is None:
        return None
    return int(round((end_nm - start_nm) / step_nm)) + 1


def make_wavelength_axis(start_nm: float, end_nm: float, n: int) -> np.ndarray:
    if n < 2 or end_nm <= start_nm:
        raise ValueError("Invalid wavelength range or number of samples")
    # Float64 keeps a picometer-scale grid numerically exact in the metadata.
    return np.linspace(start_nm, end_nm, n, dtype=np.float64)


def create_or_load_projection(
    projection_path: Path,
    compress_dim: int,
    n_pixels: int,
    seed: int,
    storage_dtype: str,
    row_block: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create/load C [compress_dim, pixels] and return its row L2 norms."""
    if projection_path.exists():
        C = np.load(projection_path, mmap_mode="r")
        if C.shape != (compress_dim, n_pixels):
            raise ValueError(
                f"Projection shape {C.shape} != expected {(compress_dim, n_pixels)}"
            )
        print(f"Loaded projection matrix: {projection_path}  shape={C.shape}")
    else:
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        dtype = np.float16 if storage_dtype == "float16" else np.float32
        print(
            f"Creating Gaussian projection C at {projection_path} "
            f"({compress_dim} x {n_pixels}, {storage_dtype})"
        )
        C_out = np.lib.format.open_memmap(
            projection_path,
            mode="w+",
            dtype=dtype,
            shape=(compress_dim, n_pixels),
        )
        rng = np.random.default_rng(seed)
        for i0 in tqdm(range(0, compress_dim, row_block), desc="Generate C"):
            i1 = min(i0 + row_block, compress_dim)
            block = rng.standard_normal((i1 - i0, n_pixels), dtype=np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            block /= np.maximum(norms, 1e-12)
            C_out[i0:i1] = block.astype(dtype, copy=False)
        C_out.flush()
        del C_out
        C = np.load(projection_path, mmap_mode="r")

    row_norms = np.empty(compress_dim, dtype=np.float32)
    for i0 in range(0, compress_dim, row_block):
        i1 = min(i0 + row_block, compress_dim)
        block = np.asarray(C[i0:i1], dtype=np.float32)
        row_norms[i0:i1] = np.linalg.norm(block, axis=1)

    if not np.isfinite(row_norms).all() or np.any(row_norms <= 0):
        raise ValueError("Projection matrix contains invalid/zero rows")
    return C, row_norms


def effective_matrix_metadata(
    args,
    T: TransmissionMatrix,
    projection_path: Path,
) -> Dict:
    column_scale = None
    if args.T_column_scale_path:
        column_scale = short_file_fingerprint(Path(args.T_column_scale_path))
    return {
        "format_version": FORMAT_VERSION,
        "T": short_file_fingerprint(Path(args.T_path)),
        "T_original_shape": list(T.array.shape),
        "T_effective_shape": list(T.shape),
        "T_layout": T.layout,
        "T_normalization": args.T_normalization,
        "T_column_scale": column_scale,
        "projection": short_file_fingerprint(projection_path),
        "projection_shape": [args.compress_dim, T.n_pixels],
        "compress_dim": args.compress_dim,
        "wavelength_start_nm": args.wl_start_nm,
        "wavelength_end_nm": args.wl_end_nm,
        "n_wavelengths": T.n_wavelengths,
    }


def build_or_load_effective_matrix(
    T: TransmissionMatrix,
    C: np.ndarray,
    args,
    device: torch.device,
    projection_path: Path,
) -> np.ndarray:
    """Construct W_eff = C @ T in wavelength blocks and cache it as .npy."""
    physics_dir = Path(args.out_dir) / "physics"
    physics_dir.mkdir(parents=True, exist_ok=True)
    w_path = physics_dir / "W_eff.npy"
    meta_path = physics_dir / "W_eff_metadata.json"
    requested_meta = effective_matrix_metadata(args, T, projection_path)

    if w_path.exists() and meta_path.exists() and not args.force_rebuild_physics:
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        if existing_meta == requested_meta:
            W = np.load(w_path, mmap_mode="r")
            expected_shape = (args.compress_dim, T.n_wavelengths)
            if W.shape != expected_shape:
                raise ValueError(f"Cached W_eff shape {W.shape} != {expected_shape}")
            print(f"Loaded cached effective matrix: {w_path}")
            return W
        raise RuntimeError(
            "Existing W_eff cache does not match the current T/projection/settings. "
            "Use a new --out_dir or pass --force_rebuild_physics."
        )

    print(f"Building W_eff = C @ T on {device} ...")
    column_scale = None
    if args.T_column_scale_path:
        column_scale = np.asarray(np.load(args.T_column_scale_path), dtype=np.float32).reshape(-1)
        if column_scale.size != T.n_wavelengths:
            raise ValueError(
                f"T column scale has {column_scale.size} values; expected {T.n_wavelengths}"
            )
        if not np.isfinite(column_scale).all() or np.any(column_scale <= 0):
            raise ValueError("T column scale must contain finite positive multipliers")

    t_bytes = T.n_pixels * T.n_wavelengths * np.dtype(np.float32).itemsize
    c_bytes = args.compress_dim * T.n_pixels * np.dtype(np.float32).itemsize
    w_bytes = args.compress_dim * T.n_wavelengths * np.dtype(np.float32).itemsize
    required_gpu_bytes = t_bytes + c_bytes + w_bytes

    build_mode = args.physics_build_mode
    if build_mode == "auto":
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            safe_bytes = int(free_bytes * args.physics_gpu_memory_fraction)
            build_mode = "full_gpu" if required_gpu_bytes <= safe_bytes else "disk_stream"
            print(
                "Physics build auto-selection: "
                f"estimated tensors={required_gpu_bytes / 2**30:.2f} GiB, "
                f"free VRAM={free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB, "
                f"selected={build_mode}"
            )
        else:
            build_mode = "disk_stream"

    if build_mode == "full_gpu" and device.type != "cuda":
        raise ValueError("--physics_build_mode full_gpu requires a CUDA device")

    # C is copied once into writable, contiguous RAM.  This removes the warning
    # caused by torch.from_numpy(read_only_memmap) and accelerates the H2D copy.
    print(f"Loading C into RAM ({c_bytes / 2**30:.2f} GiB FP32) ...")
    C_ram = np.array(C, dtype=np.float32, order="C", copy=True)
    C_device = torch.from_numpy(C_ram).to(device=device, non_blocking=False)

    T_ram = None
    if build_mode in {"full_gpu", "ram_stream"}:
        print(
            f"Loading T sequentially into RAM once ({t_bytes / 2**30:.2f} GiB FP32). "
            "This can take a few minutes but avoids repeated random disk reads ..."
        )
        load_start = time.perf_counter()
        T_ram = T.materialize_pixels_by_wavelength()
        print(f"T loaded into RAM in {(time.perf_counter() - load_start) / 60:.2f} min")
        if not np.isfinite(T_ram).all():
            raise ValueError("Non-finite values found in T")
        if column_scale is not None:
            T_ram *= column_scale[None, :]
        if args.T_normalization == "column_l2":
            # Sum by pixel blocks to avoid a second 20+ GiB temporary array.
            norm_sq = np.zeros(T.n_wavelengths, dtype=np.float64)
            for i0 in tqdm(range(0, T.n_pixels, 8192), desc="Column norms"):
                block = T_ram[i0 : i0 + 8192]
                norm_sq += np.einsum("ij,ij->j", block, block, dtype=np.float64)
            norms = np.sqrt(norm_sq).astype(np.float32)
            T_ram /= np.maximum(norms[None, :], 1e-12)
        elif args.T_normalization != "none":
            raise ValueError(f"Unknown T_normalization: {args.T_normalization}")

    W_out = np.lib.format.open_memmap(
        w_path,
        mode="w+",
        dtype=np.float32,
        shape=(args.compress_dim, T.n_wavelengths),
    )

    if build_mode == "full_gpu":
        t_device = None
        w_device = None
        try:
            print(f"Copying full T to {device} and executing one GEMM ...")
            with torch.inference_mode():
                t_device = torch.from_numpy(T_ram).to(device=device, non_blocking=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                gemm_start = time.perf_counter()
                w_device = torch.matmul(C_device, t_device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(f"Full C @ T GEMM finished in {time.perf_counter() - gemm_start:.2f} s")
                W_out[:] = w_device.float().cpu().numpy()
                del t_device, w_device
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(
                "Full-GPU build ran out of VRAM; falling back to RAM streaming. "
                "No disk reread is required."
            )
            del t_device, w_device
            if device.type == "cuda":
                torch.cuda.empty_cache()
            build_mode = "ram_stream"

    if build_mode in {"ram_stream", "disk_stream"}:
        source_label = "RAM" if T_ram is not None else "disk memmap"
        print(
            f"Building W_eff by wavelength blocks from {source_label}; "
            f"block={args.physics_wavelength_block}"
        )
        with torch.inference_mode():
            for j0 in tqdm(
                range(0, T.n_wavelengths, args.physics_wavelength_block),
                desc="Build W_eff",
            ):
                j1 = min(j0 + args.physics_wavelength_block, T.n_wavelengths)
                if T_ram is not None:
                    t_np = np.ascontiguousarray(T_ram[:, j0:j1])
                else:
                    t_np = np.ascontiguousarray(
                        T.wavelength_block(j0, j1), dtype=np.float32
                    )
                    if not np.isfinite(t_np).all():
                        raise ValueError(f"Non-finite values found in T columns [{j0}:{j1}]")
                    if column_scale is not None:
                        t_np *= column_scale[None, j0:j1]
                    if args.T_normalization == "column_l2":
                        norms = np.linalg.norm(t_np, axis=0, keepdims=True)
                        t_np /= np.maximum(norms, 1e-12)
                    elif args.T_normalization != "none":
                        raise ValueError(f"Unknown T_normalization: {args.T_normalization}")

                t_device = torch.from_numpy(t_np).to(device=device, non_blocking=False)
                w_block = torch.matmul(C_device, t_device)
                W_out[:, j0:j1] = w_block.float().cpu().numpy()
                del t_device, w_block, t_np

    W_out.flush()
    del W_out, C_device, C_ram, T_ram
    if device.type == "cuda":
        torch.cuda.empty_cache()

    save_json(meta_path, requested_meta)
    return np.load(w_path, mmap_mode="r")


# -----------------------------------------------------------------------------
# Deterministic universal synthetic dataset
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumGeneratorConfig:
    # ``single_line`` is a strict one-hot subset of the paper's sparse mode.
    # Giving it an explicit probability prevents mixed validation metrics from
    # hiding catastrophic monochromatic failures.
    single_line_probability: float = 0.30
    sparse_probability: float = 0.20
    doublet_probability: float = 0.25
    continuous_probability: float = 0.25
    sparse_min_peaks: int = 1
    sparse_max_peaks: int = 5
    continuous_min_envelopes: int = 3
    continuous_max_envelopes: int = 10
    narrow_sigma_min_nm: float = 0.003
    narrow_sigma_max_nm: float = 0.040
    doublet_sigma_min_nm: float = 0.003
    doublet_sigma_max_nm: float = 0.006
    doublet_spacing_min_nm: float = 0.010
    doublet_spacing_max_nm: float = 0.030
    broad_sigma_min_nm: float = 1.0
    broad_sigma_max_nm: float = 10.0
    polynomial_background_probability: float = 0.5

    def validate(self) -> None:
        probs = (
            self.single_line_probability,
            self.sparse_probability,
            self.doublet_probability,
            self.continuous_probability,
        )
        if any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
            raise ValueError(f"Mode probabilities must be non-negative and sum to 1, got {probs}")
        if self.doublet_spacing_min_nm <= 0:
            raise ValueError("Doublet spacing must be positive")


class UniversalSpectraDataset(Dataset):
    """Index-deterministic dataset; safe with multiple DataLoader workers."""

    def __init__(
        self,
        num_samples: int,
        wavelength_nm: np.ndarray,
        config: SpectrumGeneratorConfig,
        seed: int,
        forced_mode: Optional[str] = None,
    ):
        self.num_samples = int(num_samples)
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
        self.config = config
        self.config.validate()
        self.seed = int(seed)
        self.forced_mode = forced_mode
        valid_modes = {None, "single_line", "sparse", "doublet", "continuous"}
        if self.forced_mode not in valid_modes:
            raise ValueError(f"Unknown forced spectrum mode: {self.forced_mode!r}")
        self.M = self.wavelength_nm.size
        self.step_nm = float(np.median(np.diff(self.wavelength_nm)))

        min_bins = max(1, int(math.ceil(config.doublet_spacing_min_nm / self.step_nm - 1e-7)))
        max_bins = int(math.floor(config.doublet_spacing_max_nm / self.step_nm + 1e-7))
        if max_bins < min_bins:
            raise ValueError(
                "The wavelength grid cannot represent the requested doublet range: "
                f"grid step={self.step_nm * 1000:.3f} pm, "
                f"spacing=[{config.doublet_spacing_min_nm * 1000:.1f}, "
                f"{config.doublet_spacing_max_nm * 1000:.1f}] pm"
            )
        self.doublet_bins = (min_bins, max_bins)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        # SeedSequence avoids the repeated-worker RNG bug of a mutable RandomState.
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(idx)]))

    def _add_gaussian(
        self,
        spectrum: np.ndarray,
        center_index: int,
        sigma_nm: float,
        amplitude: float,
    ) -> None:
        sigma_nm = max(float(sigma_nm), self.step_nm * 0.25)
        half_width = max(1, int(math.ceil(4.0 * sigma_nm / self.step_nm)))
        i0 = max(0, center_index - half_width)
        i1 = min(self.M, center_index + half_width + 1)
        center_nm = float(self.wavelength_nm[center_index])
        x = self.wavelength_nm[i0:i1].astype(np.float64)
        spectrum[i0:i1] += amplitude * np.exp(-0.5 * ((x - center_nm) / sigma_nm) ** 2)

    def _sample_sparse(self, spectrum: np.ndarray, rng: np.random.Generator) -> None:
        cfg = self.config
        k = int(rng.integers(cfg.sparse_min_peaks, cfg.sparse_max_peaks + 1))
        for _ in range(k):
            center = int(rng.integers(0, self.M))
            sigma = float(rng.uniform(cfg.narrow_sigma_min_nm, cfg.narrow_sigma_max_nm))
            amplitude = float(rng.uniform(0.1, 1.0))
            self._add_gaussian(spectrum, center, sigma, amplitude)

    def _sample_single_line(
        self,
        spectrum: np.ndarray,
        rng: np.random.Generator,
        center_index: Optional[int] = None,
    ) -> None:
        """Exact one-hot line on the calibrated wavelength grid."""
        center = int(rng.integers(0, self.M)) if center_index is None else int(center_index)
        if not 0 <= center < self.M:
            raise ValueError(f"Single-line center index {center} is outside [0, {self.M})")
        spectrum[center] = 1.0

    def _sample_doublet(self, spectrum: np.ndarray, rng: np.random.Generator) -> None:
        cfg = self.config
        min_bins, max_bins = self.doublet_bins
        separation = int(rng.integers(min_bins, max_bins + 1))
        left = int(rng.integers(0, self.M - separation))
        right = left + separation
        sigma = float(rng.uniform(cfg.doublet_sigma_min_nm, cfg.doublet_sigma_max_nm))
        self._add_gaussian(spectrum, left, sigma, float(rng.uniform(0.5, 1.0)))
        self._add_gaussian(spectrum, right, sigma, float(rng.uniform(0.5, 1.0)))

    def _sample_continuous(self, spectrum: np.ndarray, rng: np.random.Generator) -> None:
        cfg = self.config
        k = int(rng.integers(cfg.continuous_min_envelopes, cfg.continuous_max_envelopes + 1))
        for _ in range(k):
            center = int(rng.integers(0, self.M))
            sigma = float(rng.uniform(cfg.broad_sigma_min_nm, cfg.broad_sigma_max_nm))
            amplitude = float(rng.uniform(0.2, 1.0))
            self._add_gaussian(spectrum, center, sigma, amplitude)

        if rng.random() < cfg.polynomial_background_probability:
            x = np.linspace(-1.0, 1.0, self.M, dtype=np.float32)
            coefficients = rng.normal(0.0, 0.01, size=3)
            spectrum += np.abs(np.polyval(coefficients, x)).astype(np.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = self._rng(idx)
        spectrum = np.zeros(self.M, dtype=np.float32)
        mode = self.forced_mode
        single_line_center: Optional[int] = None
        if mode is None:
            # Contiguous deterministic partitions give exact mode counts.  The
            # shuffled DataLoader still mixes modes within mini-batches.
            r = (int(idx) + 0.5) / self.num_samples
            p_single = self.config.single_line_probability
            p_sparse = self.config.sparse_probability
            p_doublet = self.config.doublet_probability
            if r < p_single:
                mode = "single_line"
                # The first single-line partition cycles through every
                # wavelength grid point, guaranteeing complete coverage.
                single_line_center = int(idx) % self.M
            elif r < p_single + p_sparse:
                mode = "sparse"
            elif r < p_single + p_sparse + p_doublet:
                mode = "doublet"
            else:
                mode = "continuous"
        elif mode == "single_line":
            # A dedicated validation set spans the complete wavelength range
            # uniformly instead of sampling only a local/random subset.
            if self.num_samples <= 1:
                single_line_center = self.M // 2
            else:
                single_line_center = int(
                    round(int(idx) * (self.M - 1) / (self.num_samples - 1))
                )

        if mode == "single_line":
            self._sample_single_line(spectrum, rng, center_index=single_line_center)
        elif mode == "sparse":
            self._sample_sparse(spectrum, rng)
        elif mode == "doublet":
            self._sample_doublet(spectrum, rng)
        elif mode == "continuous":
            self._sample_continuous(spectrum, rng)
        else:
            raise RuntimeError(f"Unhandled spectrum mode: {mode!r}")

        np.maximum(spectrum, 0.0, out=spectrum)
        peak = float(spectrum.max())
        if not np.isfinite(peak) or peak <= 1e-12:
            raise RuntimeError(f"Invalid synthetic spectrum at index {idx}")
        spectrum /= peak
        return torch.from_numpy(spectrum)


# -----------------------------------------------------------------------------
# Fixed physical encoder and trainable decoder
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Pre-normalized residual MLP block; its last layer starts at zero."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))
        return x + residual


class SpectralDecoder(nn.Module):
    """Compressed measurement -> non-negative, peak-normalized relative spectrum."""

    def __init__(
        self,
        compress_dim: int,
        spectrum_dim: int,
        hidden_dim: int = 1536,
        num_res_blocks: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.compress_dim = int(compress_dim)
        self.spectrum_dim = int(spectrum_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.dropout = float(dropout)

        # LayerNorm avoids train/test running-statistic mismatch and is suitable
        # because all target spectra are peak-normalized relative spectra.
        self.input_norm = nn.LayerNorm(compress_dim)
        self.head = nn.Sequential(nn.Linear(compress_dim, hidden_dim), nn.GELU())
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, spectrum_dim))
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        head_linear = self.head[0]
        tail_linear = self.tail[1]
        nn.init.xavier_uniform_(head_linear.weight)
        nn.init.zeros_(head_linear.bias)
        nn.init.xavier_uniform_(tail_linear.weight, gain=0.1)
        nn.init.constant_(tail_linear.bias, -4.0)

    def config(self) -> Dict:
        return {
            "compress_dim": self.compress_dim,
            "spectrum_dim": self.spectrum_dim,
            "hidden_dim": self.hidden_dim,
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
            "input_normalization": "LayerNorm",
            "output_activation": "Softplus",
            "output_normalization": "per-spectrum peak (L-infinity)",
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(z)
        x = self.head(x)
        x = self.body(x)
        spectrum = self.softplus(self.tail(x))
        peak = spectrum.amax(dim=1, keepdim=True).clamp_min(1e-8)
        return spectrum / peak


class PhysicsInformedAutoencoder(nn.Module):
    """Training wrapper.  Only ``decoder`` contains trainable parameters."""

    def __init__(
        self,
        W_eff: torch.Tensor,
        projection_row_norms: torch.Tensor,
        decoder: SpectralDecoder,
        noise_model: str,
        noise_std_min: float,
        noise_std_max: float,
        relative_noise_std_min: float = 0.0,
        relative_noise_std_max: float = 0.0,
        encoder_drift_std_max: float = 0.0,
        gain_jitter: float = 0.0,
    ):
        super().__init__()
        if W_eff.ndim != 2:
            raise ValueError("W_eff must have shape [compress_dim, spectrum_dim]")
        if W_eff.shape[0] != projection_row_norms.numel():
            raise ValueError("projection_row_norms has the wrong length")

        # persistent=False keeps the large fixed matrix out of decoder checkpoints.
        self.register_buffer("W_eff", W_eff, persistent=False)
        self.register_buffer("projection_row_norms", projection_row_norms, persistent=False)
        self.decoder = decoder
        self.noise_model = noise_model
        self.noise_std_min = float(noise_std_min)
        self.noise_std_max = float(noise_std_max)
        self.relative_noise_std_min = float(relative_noise_std_min)
        self.relative_noise_std_max = float(relative_noise_std_max)
        self.encoder_drift_std_max = float(encoder_drift_std_max)
        self.gain_jitter = float(gain_jitter)

    def encode_clean(self, spectrum: torch.Tensor) -> torch.Tensor:
        return F.linear(spectrum, self.W_eff)

    def add_measurement_noise(self, z: torch.Tensor, force: bool = False) -> torch.Tensor:
        noise_enabled = (
            self.noise_std_max > 0
            or self.relative_noise_std_max > 0
            or self.encoder_drift_std_max > 0
            or self.gain_jitter > 0
        )
        if (not self.training and not force) or not noise_enabled:
            return z

        batch_size = z.shape[0]
        noisy = z
        if self.noise_model in {"projected_pixel_awgn", "hybrid_awgn"} and self.noise_std_max > 0:
            sigma = torch.empty(batch_size, 1, device=z.device, dtype=z.dtype)
            sigma.uniform_(self.noise_std_min, self.noise_std_max)
            # Diagonal covariance approximation:
            # Std[(C eta)_i] = sigma_pixel * ||C_i||_2.
            scale = self.projection_row_norms.to(dtype=z.dtype).unsqueeze(0)
            noisy = noisy + torch.randn_like(z) * sigma * scale
        elif self.noise_model not in {"relative_encoder_awgn", "hybrid_awgn"}:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

        z_rms = z.detach().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        if self.noise_model in {"relative_encoder_awgn", "hybrid_awgn"} and self.relative_noise_std_max > 0:
            relative_sigma = torch.empty(batch_size, 1, device=z.device, dtype=z.dtype)
            relative_sigma.uniform_(self.relative_noise_std_min, self.relative_noise_std_max)
            noisy = noisy + torch.randn_like(z) * relative_sigma * z_rms

        if self.encoder_drift_std_max > 0:
            # One random drift direction is shared within a mini-batch.  Across
            # batches its direction changes, approximating slowly varying
            # camera/background mismatch without memorizing a fixed offset.
            direction = torch.randn(1, z.shape[1], device=z.device, dtype=z.dtype)
            direction = direction / direction.square().mean().sqrt().clamp_min(1e-8)
            drift_sigma = torch.empty(batch_size, 1, device=z.device, dtype=z.dtype)
            drift_sigma.uniform_(0.0, self.encoder_drift_std_max)
            noisy = noisy + direction * drift_sigma * z_rms

        if self.gain_jitter > 0:
            gain = torch.empty(batch_size, 1, device=z.device, dtype=z.dtype)
            gain.uniform_(1.0 - self.gain_jitter, 1.0 + self.gain_jitter)
            noisy = noisy * gain
        return noisy

    def forward(
        self,
        spectrum: torch.Tensor,
        inject_noise: Optional[bool] = None,
    ) -> torch.Tensor:
        z = self.encode_clean(spectrum)
        if inject_noise is None:
            inject_noise = self.training
        if inject_noise:
            z = self.add_measurement_noise(z, force=True)
        return self.decoder(z)


# -----------------------------------------------------------------------------
# Loss, evaluation, visualization, and checkpointing
# -----------------------------------------------------------------------------


def composite_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.1,
    peak_weight: float = 20.0,
    emd_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Keep all reductions in FP32 even when the network uses mixed precision.
    rec = reconstruction.float()
    tar = target.float()

    rec_norm = F.normalize(rec, p=2, dim=1, eps=1e-8)
    tar_norm = F.normalize(tar, p=2, dim=1, eps=1e-8)
    loss_cos = 1.0 - torch.sum(rec_norm * tar_norm, dim=1).mean()

    weights = 1.0 + peak_weight * tar
    loss_amp = torch.mean(weights * torch.abs(rec - tar))

    loss_grad = F.l1_loss(rec[:, 1:] - rec[:, :-1], tar[:, 1:] - tar[:, :-1])

    # 1-D Wasserstein/EMD surrogate.  Unlike cosine loss, this term knows that
    # a peak displaced by 100 nm is worse than one displaced by a single bin.
    rec_pdf = rec / rec.sum(dim=1, keepdim=True).clamp_min(1e-8)
    tar_pdf = tar / tar.sum(dim=1, keepdim=True).clamp_min(1e-8)
    loss_emd = torch.mean(
        torch.abs(torch.cumsum(rec_pdf, dim=1) - torch.cumsum(tar_pdf, dim=1))
    )

    total = alpha * loss_cos + beta * loss_amp + gamma * loss_grad + emd_weight * loss_emd
    return total, {
        "cos": loss_cos,
        "amp": loss_amp,
        "grad": loss_grad,
        "emd": loss_emd,
    }


def two_view_consistency_loss(
    reconstruction_a: torch.Tensor,
    reconstruction_b: torch.Tensor,
) -> torch.Tensor:
    """Require two independent noisy measurements of one spectrum to agree."""
    a = reconstruction_a.float()
    b = reconstruction_b.float()
    loss_l1 = F.l1_loss(a, b)
    loss_cos = 1.0 - F.cosine_similarity(a, b, dim=1, eps=1e-8).mean()
    return loss_l1 + loss_cos


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    loss_weights: Tuple[float, float, float, float, float],
    inject_noise: bool = False,
    validation_seed: int = 0,
    peak_step_nm: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    sums = {
        "loss": 0.0,
        "cos": 0.0,
        "amp": 0.0,
        "grad": 0.0,
        "emd": 0.0,
        "fidelity": 0.0,
    }
    peak_abs_error_nm = 0.0
    peak_squared_error_nm = 0.0
    peak_within_1bin = 0
    peak_within_2bin = 0
    peak_within_5bin = 0
    count = 0

    cuda_devices = None
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    # The same held-out spectra receive the same noise realization every epoch.
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(validation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(validation_seed)
        for target in loader:
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None),
            ):
                reconstruction = model(target, inject_noise=inject_noise)
            loss, parts = composite_loss(reconstruction, target, *loss_weights)
            fidelity = F.cosine_similarity(reconstruction.float(), target.float(), dim=1).mean()
            n = target.shape[0]
            sums["loss"] += float(loss) * n
            sums["cos"] += float(parts["cos"]) * n
            sums["amp"] += float(parts["amp"]) * n
            sums["grad"] += float(parts["grad"]) * n
            sums["emd"] += float(parts["emd"]) * n
            sums["fidelity"] += float(fidelity) * n
            if peak_step_nm is not None:
                target_indices = target.argmax(dim=1)
                reconstruction_indices = reconstruction.argmax(dim=1)
                error_bins = (reconstruction_indices - target_indices).abs()
                error_nm = error_bins.float() * peak_step_nm
                peak_abs_error_nm += float(error_nm.sum())
                peak_squared_error_nm += float((error_nm * error_nm).sum())
                peak_within_1bin += int((error_bins <= 1).sum())
                peak_within_2bin += int((error_bins <= 2).sum())
                peak_within_5bin += int((error_bins <= 5).sum())
            count += n

    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if peak_step_nm is not None:
        metrics.update(
            {
                "peak_mae_nm": peak_abs_error_nm / max(count, 1),
                "peak_rmse_nm": math.sqrt(peak_squared_error_nm / max(count, 1)),
                "peak_accuracy_within_1bin": peak_within_1bin / max(count, 1),
                "peak_accuracy_within_2bin": peak_within_2bin / max(count, 1),
                "peak_accuracy_within_5bin": peak_within_5bin / max(count, 1),
            }
        )
    return metrics


@torch.inference_mode()
def evaluate_full_ideal_single_line_scan(
    decoder: nn.Module,
    W_eff: torch.Tensor,
    wavelength_nm: np.ndarray,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    batch_size: int,
    figure_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Decode every ideal W_eff column and report catastrophic outliers."""
    decoder.eval()
    if batch_size <= 0:
        raise ValueError("ideal_scan_batch_size must be positive")
    n_wavelengths = W_eff.shape[1]
    predicted_indices = np.empty(n_wavelengths, dtype=np.int64)
    for i0 in range(0, n_wavelengths, batch_size):
        i1 = min(i0 + batch_size, n_wavelengths)
        z = W_eff[:, i0:i1].T
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None),
        ):
            reconstruction = decoder(z)
        predicted_indices[i0:i1] = reconstruction.argmax(dim=1).cpu().numpy()

    true_indices = np.arange(n_wavelengths, dtype=np.int64)
    step_nm = float(np.median(np.diff(wavelength_nm)))
    error_nm = (predicted_indices - true_indices).astype(np.float64) * step_nm
    abs_error_nm = np.abs(error_nm)
    metrics = {
        "mae_nm": float(abs_error_nm.mean()),
        "median_absolute_error_nm": float(np.median(abs_error_nm)),
        "rmse_nm": float(np.sqrt(np.mean(error_nm * error_nm))),
        "max_absolute_error_nm": float(abs_error_nm.max()),
        "accuracy_within_1bin": float(np.mean(abs_error_nm <= step_nm + 1e-12)),
        "accuracy_within_2bin": float(np.mean(abs_error_nm <= 2 * step_nm + 1e-12)),
        "accuracy_within_5bin": float(np.mean(abs_error_nm <= 5 * step_nm + 1e-12)),
        "catastrophic_over_1nm_count": int(np.sum(abs_error_nm > 1.0)),
    }

    if figure_path is not None:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(wavelength_nm, error_nm * 1000.0, color="#1f77b4", lw=0.7)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(50.0, color="#d62728", lw=0.8, ls="--")
        ax.axhline(-50.0, color="#d62728", lw=0.8, ls="--")
        ax.set_xlabel("True wavelength (nm)")
        ax.set_ylabel("Ideal-input peak error (pm)")
        ax.set_title(
            f"Ideal scan | MAE={metrics['mae_nm'] * 1000:.2f} pm | "
            f"within 5 bins={metrics['accuracy_within_5bin'] * 100:.3f}%"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
    return metrics


@torch.inference_mode()
def save_reconstruction_examples(
    model: nn.Module,
    samples: torch.Tensor,
    wavelength_nm: np.ndarray,
    device: torch.device,
    path: Path,
    amp_dtype: Optional[torch.dtype],
) -> None:
    model.eval()
    target = samples.to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(amp_dtype is not None),
    ):
        rec = model(target).float()
    rec_np = rec.cpu().numpy()
    target_np = target.cpu().numpy()
    fidelities = np.sum(rec_np * target_np, axis=1) / (
        np.linalg.norm(rec_np, axis=1) * np.linalg.norm(target_np, axis=1) + 1e-12
    )

    n = target_np.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(wavelength_nm, target_np[i], color="black", lw=1.2, label="Target")
        ax.plot(wavelength_nm, rec_np[i], color="#d62728", lw=1.0, ls="--", label="Reconstruction")
        ax.set_ylabel("Norm. intensity")
        ax.set_title(f"Validation sample {i + 1} | cosine fidelity={fidelities[i]:.5f}")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(frameon=False)
    axes[-1, 0].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def checkpoint_payload(
    model: PhysicsInformedAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    train_loader_generator: torch.Generator,
    args,
    wavelength_nm: np.ndarray,
    generator_config: SpectrumGeneratorConfig,
    metrics: Dict[str, float],
) -> Dict:
    return {
        "format_version": FORMAT_VERSION,
        "epoch": epoch,
        "decoder_state_dict": model.decoder.state_dict(),
        "decoder_config": model.decoder.config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_selection_score": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "train_loader_generator_state": train_loader_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "wavelength_nm": wavelength_nm,
        "spectrum_generator_config": asdict(generator_config),
        "training_arguments": vars(args),
        "metrics": metrics,
    }


def decoder_release_payload(
    model: PhysicsInformedAutoencoder,
    args,
    wavelength_nm: np.ndarray,
    generator_config: SpectrumGeneratorConfig,
    epoch: int,
    metrics: Dict[str, float],
    projection_path: Path,
) -> Dict:
    """Small, inference-facing checkpoint without optimizer or W_eff."""
    return {
        "format_version": FORMAT_VERSION,
        "model_type": "SpectralDecoder",
        "decoder_state_dict": model.decoder.state_dict(),
        "decoder_config": model.decoder.config(),
        "wavelength_nm": wavelength_nm,
        "spectrum_generator_config": asdict(generator_config),
        "training_noise": {
            "model": args.noise_model,
            "pixel_std_min_counts": args.noise_std_min,
            "pixel_std_max_counts": args.noise_std_max,
            "relative_std_min": args.relative_noise_std_min,
            "relative_std_max": args.relative_noise_std_max,
            "encoder_drift_std_max": args.encoder_drift_std_max,
            "gain_jitter": args.gain_jitter,
        },
        "robust_training": {
            "two_view_consistency_weight": args.consistency_weight,
            "emd_weight": args.loss_emd_weight,
            "best_model_selection": "mixed noisy loss + single-line noisy loss + peak penalty",
        },
        "physics_files": {
            "projection_matrix": str(projection_path.resolve()),
            "effective_matrix": "physics/W_eff.npy",
            "effective_matrix_metadata": "physics/W_eff_metadata.json",
            "wavelength_axis": "physics/wavelength_nm.npy",
        },
        "T_normalization": args.T_normalization,
        "T_column_scale_path": (
            str(Path(args.T_column_scale_path).resolve()) if args.T_column_scale_path else None
        ),
        "target_normalization": "per-spectrum peak normalization",
        "epoch": epoch,
        "metrics": metrics,
    }


def restore_checkpoint(
    path: Path,
    model: PhysicsInformedAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler,
    train_loader_generator: torch.Generator,
    device: torch.device,
) -> Tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"Resume checkpoint format {checkpoint.get('format_version')} != required "
            f"robust format {FORMAT_VERSION}. Start this robust run from scratch."
        )
    if checkpoint["decoder_config"] != model.decoder.config():
        raise ValueError("Resume checkpoint decoder configuration does not match this run")
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    train_loader_generator.set_state(checkpoint["train_loader_generator_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_selection_score", checkpoint["best_val_loss"])),
        int(checkpoint["epochs_without_improvement"]),
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(shuffle and len(dataset) >= batch_size),
        generator=generator,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def choose_amp_dtype(device: torch.device, amp: str) -> Optional[torch.dtype]:
    if device.type != "cuda" or amp == "none":
        return None
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use --amp fp16 or --amp none")
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(amp)


def make_optimizer(model: nn.Module, args, device: torch.device):
    kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.fused_adamw and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except (TypeError, RuntimeError) as exc:
        if "fused" not in kwargs:
            raise
        print(f"Fused AdamW unavailable ({exc}); falling back to standard AdamW.")
        kwargs.pop("fused")
        return torch.optim.AdamW(model.parameters(), **kwargs)


def write_history_row(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_arguments(args) -> None:
    if args.wl_end_nm <= args.wl_start_nm:
        raise ValueError("wl_end_nm must be larger than wl_start_nm")
    if args.wl_step_nm is not None and args.wl_step_nm <= 0:
        raise ValueError("wl_step_nm must be positive")
    if args.compress_dim <= 0 or args.hidden_dim <= 0 or args.num_res_blocks <= 0:
        raise ValueError("Network dimensions and block count must be positive")
    if args.physics_wavelength_block <= 0:
        raise ValueError("physics_wavelength_block must be positive")
    if not 0.1 <= args.physics_gpu_memory_fraction <= 0.95:
        raise ValueError("physics_gpu_memory_fraction must be in [0.1, 0.95]")
    if args.num_train_samples <= 0 or args.num_val_samples <= 0 or args.num_mode_val_samples <= 0:
        raise ValueError("Training and validation sample counts must be positive")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if args.num_viz_samples <= 0:
        raise ValueError("num_viz_samples must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 <= args.noise_std_min <= args.noise_std_max:
        raise ValueError("Require 0 <= noise_std_min <= noise_std_max")
    if not 0.0 <= args.relative_noise_std_min <= args.relative_noise_std_max:
        raise ValueError("Require 0 <= relative_noise_std_min <= relative_noise_std_max")
    if args.encoder_drift_std_max < 0:
        raise ValueError("encoder_drift_std_max must be non-negative")
    if not 0.0 <= args.gain_jitter < 1.0:
        raise ValueError("gain_jitter must be in [0, 1)")
    if args.loss_emd_weight < 0 or args.consistency_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.ideal_scan_every < 0 or args.ideal_scan_batch_size <= 0:
        raise ValueError("Require ideal_scan_every >= 0 and ideal_scan_batch_size > 0")
    if args.selection_single_weight < 0 or args.selection_peak_penalty < 0:
        raise ValueError("Model-selection weights must be non-negative")
    for low_name, high_name in (
        ("narrow_sigma_min_nm", "narrow_sigma_max_nm"),
        ("doublet_sigma_min_nm", "doublet_sigma_max_nm"),
        ("doublet_spacing_min_nm", "doublet_spacing_max_nm"),
        ("broad_sigma_min_nm", "broad_sigma_max_nm"),
    ):
        low, high = getattr(args, low_name), getattr(args, high_name)
        if low <= 0 or high < low:
            raise ValueError(f"Require 0 < {low_name} <= {high_name}")


def main(args) -> None:
    validate_arguments(args)
    seed_everything(args.seed, args.deterministic)
    out_dir = Path(args.out_dir)
    physics_dir = out_dir / "physics"
    checkpoints_dir = out_dir / "checkpoints"
    figures_dir = out_dir / "figures"
    for directory in (out_dir, physics_dir, checkpoints_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if (checkpoints_dir / "training_latest.pt").exists() and not args.resume:
        raise FileExistsError(
            f"A training checkpoint already exists in {checkpoints_dir}. "
            "Pass --resume checkpoints/training_latest.pt or choose a new --out_dir."
        )

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}) but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device.index)
    print(f"Device: {device}")

    expected_m = expected_grid_size(args.wl_start_nm, args.wl_end_nm, args.wl_step_nm)
    T = TransmissionMatrix(
        args.T_path,
        layout=args.T_layout,
        npz_key=args.T_npz_key,
        expected_wavelengths=expected_m,
    )
    if expected_m is not None and T.n_wavelengths != expected_m:
        raise ValueError(
            f"T has {T.n_wavelengths} wavelength columns, but range/step imply {expected_m}. "
            "Correct --wl_step_nm or --T_layout."
        )

    wavelength_nm = make_wavelength_axis(args.wl_start_nm, args.wl_end_nm, T.n_wavelengths)
    actual_step_nm = float(np.median(np.diff(wavelength_nm)))
    print(
        f"T effective shape: pixels={T.n_pixels}, wavelengths={T.n_wavelengths}; "
        f"grid step={actual_step_nm * 1000:.6f} pm; layout={T.layout}"
    )
    print_transmission_matrix_statistics(T)
    if args.wl_step_nm is None:
        print("Note: wavelength step was inferred from range and T column count.")

    np.save(physics_dir / "wavelength_nm.npy", wavelength_nm)
    projection_path = Path(args.projection_path) if args.projection_path else physics_dir / "projection_matrix.npy"
    C, projection_row_norms = create_or_load_projection(
        projection_path=projection_path,
        compress_dim=args.compress_dim,
        n_pixels=T.n_pixels,
        seed=args.projection_seed,
        storage_dtype=args.projection_storage_dtype,
    )
    W_eff_np = build_or_load_effective_matrix(T, C, args, device, projection_path)

    generator_cfg = SpectrumGeneratorConfig(
        single_line_probability=args.prob_single_line,
        sparse_probability=args.prob_sparse,
        doublet_probability=args.prob_doublet,
        continuous_probability=args.prob_continuous,
        sparse_min_peaks=1,
        sparse_max_peaks=5,
        continuous_min_envelopes=3,
        continuous_max_envelopes=10,
        narrow_sigma_min_nm=args.narrow_sigma_min_nm,
        narrow_sigma_max_nm=args.narrow_sigma_max_nm,
        doublet_sigma_min_nm=args.doublet_sigma_min_nm,
        doublet_sigma_max_nm=args.doublet_sigma_max_nm,
        doublet_spacing_min_nm=args.doublet_spacing_min_nm,
        doublet_spacing_max_nm=args.doublet_spacing_max_nm,
        broad_sigma_min_nm=args.broad_sigma_min_nm,
        broad_sigma_max_nm=args.broad_sigma_max_nm,
        polynomial_background_probability=args.polynomial_background_probability,
    )
    train_dataset = UniversalSpectraDataset(
        args.num_train_samples, wavelength_nm, generator_cfg, seed=args.seed
    )
    val_dataset = UniversalSpectraDataset(
        args.num_val_samples, wavelength_nm, generator_cfg, seed=args.seed + 1_000_003
    )
    mode_val_datasets = {
        mode: UniversalSpectraDataset(
            args.num_mode_val_samples,
            wavelength_nm,
            generator_cfg,
            seed=args.seed + 2_000_003 + mode_index * 100_003,
            forced_mode=mode,
        )
        for mode_index, mode in enumerate(
            ("single_line", "sparse", "doublet", "continuous")
        )
    }

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 17)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=None,
        device=device,
    )
    mode_val_loaders = {
        mode: make_loader(
            dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            generator=None,
            device=device,
        )
        for mode, dataset in mode_val_datasets.items()
    }

    # A writable copy avoids PyTorch's warning for read-only NumPy memmaps.
    W_eff = torch.from_numpy(
        np.array(W_eff_np, dtype=np.float32, order="C", copy=True)
    ).to(device)
    row_norms = torch.from_numpy(projection_row_norms).to(device)
    decoder = SpectralDecoder(
        compress_dim=args.compress_dim,
        spectrum_dim=T.n_wavelengths,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)
    model = PhysicsInformedAutoencoder(
        W_eff=W_eff,
        projection_row_norms=row_norms,
        decoder=decoder,
        noise_model=args.noise_model,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        relative_noise_std_min=args.relative_noise_std_min,
        relative_noise_std_max=args.relative_noise_std_max,
        encoder_drift_std_max=args.encoder_drift_std_max,
        gain_jitter=args.gain_jitter,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable decoder parameters: {trainable:,}")
    optimizer = make_optimizer(model.decoder, args, device)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    amp_dtype = choose_amp_dtype(device, args.amp)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )
    loss_weights = (
        args.loss_alpha,
        args.loss_beta,
        args.loss_gamma,
        args.peak_weight,
        args.loss_emd_weight,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        start_epoch, best_val_loss, epochs_without_improvement = restore_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            scaler,
            train_generator,
            device,
        )
        print(f"Resumed at epoch {start_epoch}; best validation loss={best_val_loss:.6g}")

    run_manifest = {
        "format_version": FORMAT_VERSION,
        "arguments": vars(args),
        "environment": environment_info(),
        "T_effective_shape": list(T.shape),
        "T_layout": T.layout,
        "wavelength_step_nm": actual_step_nm,
        "wavelength_step_pm": actual_step_nm * 1000.0,
        "spectrum_generator": asdict(generator_cfg),
        "decoder": model.decoder.config(),
        "trainable_parameters": trainable,
        "physics_files": {
            "projection_matrix": str(projection_path.resolve()),
            "effective_matrix": str((physics_dir / "W_eff.npy").resolve()),
            "wavelength_axis": str((physics_dir / "wavelength_nm.npy").resolve()),
        },
    }
    save_json(out_dir / "run_manifest.json", run_manifest)
    save_json(out_dir / "environment.json", run_manifest["environment"])

    mode_order = ("single_line", "sparse", "doublet", "continuous")
    validation_examples = torch.stack(
        [
            mode_val_datasets[mode_order[i % len(mode_order)]][i // len(mode_order)]
            for i in range(args.num_viz_samples)
        ]
    )
    history_path = out_dir / "history.csv"
    print(
        f"Training: {args.num_train_samples} fixed unique spectra, "
        f"{args.num_val_samples} held-out spectra, batch={args.batch_size}, epochs={args.epochs}"
    )
    print(
        "Spectrum mixture: "
        f"single-line={args.prob_single_line:.2f}, sparse={args.prob_sparse:.2f}, "
        f"doublet={args.prob_doublet:.2f}, continuous={args.prob_continuous:.2f}"
    )
    print(
        "Robust noise: "
        f"model={args.noise_model}, pixel_sigma=[{args.noise_std_min:.3f}, "
        f"{args.noise_std_max:.3f}] counts, relative_sigma=[{args.relative_noise_std_min:.3f}, "
        f"{args.relative_noise_std_max:.3f}], drift_max={args.encoder_drift_std_max:.3f}, "
        f"gain_jitter={args.gain_jitter:.3f}"
    )

    latest_ideal_scan_metrics: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = {
            "loss": 0.0,
            "supervised": 0.0,
            "consistency": 0.0,
            "cos": 0.0,
            "amp": 0.0,
            "grad": 0.0,
            "emd": 0.0,
        }
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)
        for target in progress:
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None),
            ):
                z_clean = model.encode_clean(target)
                z_view_a = model.add_measurement_noise(z_clean, force=True)
                z_view_b = model.add_measurement_noise(z_clean, force=True)
                reconstruction_both = model.decoder(torch.cat((z_view_a, z_view_b), dim=0))
                reconstruction_a, reconstruction_b = reconstruction_both.chunk(2, dim=0)
            supervised_loss, parts = composite_loss(
                reconstruction_both,
                torch.cat((target, target), dim=0),
                *loss_weights,
            )
            consistency_loss = two_view_consistency_loss(
                reconstruction_a,
                reconstruction_b,
            )
            loss = supervised_loss + args.consistency_weight * consistency_loss

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            n = target.shape[0]
            running["loss"] += float(loss.detach()) * n
            running["supervised"] += float(supervised_loss.detach()) * n
            running["consistency"] += float(consistency_loss.detach()) * n
            for key in ("cos", "amp", "grad", "emd"):
                running[key] += float(parts[key].detach()) * n
            seen += n
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        train_metrics = {key: value / max(seen, 1) for key, value in running.items()}
        val_clean = evaluate(
            model,
            val_loader,
            device,
            amp_dtype,
            loss_weights,
            inject_noise=False,
            validation_seed=args.seed + 91,
        )
        val_noisy = evaluate(
            model,
            val_loader,
            device,
            amp_dtype,
            loss_weights,
            inject_noise=True,
            validation_seed=args.seed + 92,
        )

        mode_noisy: Dict[str, Dict[str, float]] = {}
        for mode_index, (mode, loader) in enumerate(mode_val_loaders.items()):
            mode_noisy[mode] = evaluate(
                model,
                loader,
                device,
                amp_dtype,
                loss_weights,
                inject_noise=True,
                validation_seed=args.seed + 200 + mode_index,
                peak_step_nm=(actual_step_nm if mode == "single_line" else None),
            )
        single_clean = evaluate(
            model,
            mode_val_loaders["single_line"],
            device,
            amp_dtype,
            loss_weights,
            inject_noise=False,
            validation_seed=args.seed + 299,
            peak_step_nm=actual_step_nm,
        )

        if (
            not latest_ideal_scan_metrics
            or epoch == 1
            or (args.ideal_scan_every > 0 and epoch % args.ideal_scan_every == 0)
            or epoch == args.epochs
        ):
            latest_ideal_scan_metrics = evaluate_full_ideal_single_line_scan(
                decoder=model.decoder,
                W_eff=model.W_eff,
                wavelength_nm=wavelength_nm,
                device=device,
                amp_dtype=amp_dtype,
                batch_size=args.ideal_scan_batch_size,
                figure_path=figures_dir / f"ideal_scan_epoch_{epoch:03d}.png",
            )
            save_json(
                figures_dir / f"ideal_scan_epoch_{epoch:03d}.json",
                latest_ideal_scan_metrics,
            )

        single_noisy = mode_noisy["single_line"]
        selection_score = (
            val_noisy["loss"]
            + args.selection_single_weight * single_noisy["loss"]
            + args.selection_peak_penalty
            * (1.0 - single_noisy["peak_accuracy_within_2bin"])
        )
        scheduler.step(selection_score)
        epoch_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_supervised": train_metrics["supervised"],
            "train_consistency": train_metrics["consistency"],
            "train_cos": train_metrics["cos"],
            "train_amp": train_metrics["amp"],
            "train_grad": train_metrics["grad"],
            "train_emd": train_metrics["emd"],
            "val_clean_loss": val_clean["loss"],
            "val_clean_fidelity": val_clean["fidelity"],
            "val_noisy_loss": val_noisy["loss"],
            "val_noisy_cos": val_noisy["cos"],
            "val_noisy_amp": val_noisy["amp"],
            "val_noisy_grad": val_noisy["grad"],
            "val_noisy_emd": val_noisy["emd"],
            "val_noisy_fidelity": val_noisy["fidelity"],
            "val_single_clean_loss": single_clean["loss"],
            "val_single_clean_peak_mae_pm": single_clean["peak_mae_nm"] * 1000.0,
            "val_single_clean_accuracy_5bin": single_clean["peak_accuracy_within_5bin"],
            "val_single_noisy_loss": single_noisy["loss"],
            "val_single_noisy_fidelity": single_noisy["fidelity"],
            "val_single_noisy_peak_mae_pm": single_noisy["peak_mae_nm"] * 1000.0,
            "val_single_noisy_peak_rmse_pm": single_noisy["peak_rmse_nm"] * 1000.0,
            "val_single_noisy_accuracy_1bin": single_noisy["peak_accuracy_within_1bin"],
            "val_single_noisy_accuracy_2bin": single_noisy["peak_accuracy_within_2bin"],
            "val_single_noisy_accuracy_5bin": single_noisy["peak_accuracy_within_5bin"],
            "val_sparse_noisy_loss": mode_noisy["sparse"]["loss"],
            "val_sparse_noisy_fidelity": mode_noisy["sparse"]["fidelity"],
            "val_doublet_noisy_loss": mode_noisy["doublet"]["loss"],
            "val_doublet_noisy_fidelity": mode_noisy["doublet"]["fidelity"],
            "val_continuous_noisy_loss": mode_noisy["continuous"]["loss"],
            "val_continuous_noisy_fidelity": mode_noisy["continuous"]["fidelity"],
            "ideal_scan_mae_pm": latest_ideal_scan_metrics.get("mae_nm", float("nan")) * 1000.0,
            "ideal_scan_accuracy_5bin": latest_ideal_scan_metrics.get(
                "accuracy_within_5bin", float("nan")
            ),
            "ideal_scan_catastrophic_over_1nm": latest_ideal_scan_metrics.get(
                "catastrophic_over_1nm_count", float("nan")
            ),
            "selection_score": selection_score,
            "learning_rate": lr,
            "epoch_seconds": epoch_seconds,
        }
        write_history_row(history_path, row)
        print(
            f"[Epoch {epoch:03d}] train={train_metrics['loss']:.6f} | "
            f"val_clean={val_clean['loss']:.6f} | val_noisy={val_noisy['loss']:.6f} | "
            f"single_noisy={single_noisy['loss']:.6f} | "
            f"single_MAE={single_noisy['peak_mae_nm'] * 1000:.2f} pm | "
            f"single_acc20={single_noisy['peak_accuracy_within_2bin'] * 100:.2f}% | "
            f"score={selection_score:.6f} | "
            f"lr={lr:.3e} | {epoch_seconds / 60:.2f} min"
        )

        metrics_payload = {
            "mixed_clean": val_clean,
            "mixed_noisy": val_noisy,
            "single_clean": single_clean,
            "mode_noisy": mode_noisy,
            "ideal_scan": latest_ideal_scan_metrics,
            "selection_score": selection_score,
        }
        improved = selection_score < best_val_loss - args.early_stop_min_delta
        if improved:
            best_val_loss = selection_score
            epochs_without_improvement = 0
            best_payload = decoder_release_payload(
                model,
                args,
                wavelength_nm,
                generator_cfg,
                epoch,
                metrics_payload,
                projection_path,
            )
            atomic_torch_save(best_payload, checkpoints_dir / "decoder_best.pt")
        else:
            epochs_without_improvement += 1

        latest_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            epochs_without_improvement,
            train_generator,
            args,
            wavelength_nm,
            generator_cfg,
            metrics_payload,
        )
        atomic_torch_save(latest_payload, checkpoints_dir / "training_latest.pt")
        atomic_torch_save(
            decoder_release_payload(
                model,
                args,
                wavelength_nm,
                generator_cfg,
                epoch,
                metrics_payload,
                projection_path,
            ),
            checkpoints_dir / "decoder_latest.pt",
        )

        if epoch == 1 or epoch % args.viz_every == 0 or epoch == args.epochs:
            save_reconstruction_examples(
                model,
                validation_examples,
                wavelength_nm,
                device,
                figures_dir / f"validation_epoch_{epoch:03d}.png",
                amp_dtype,
            )

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping: no validation improvement for "
                f"{epochs_without_improvement} epochs."
            )
            break

    print(f"Finished. Best decoder: {checkpoints_dir / 'decoder_best.pt'}")
    print("The experimental inference code must use: speckle -> saved C -> saved decoder.")


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train the fixed-physics encoder + ResNet spectral decoder.",
    )

    # Physical system.
    p.add_argument("--T_path", default=str(DEFAULT_T_PATH), help="Experimental T matrix (.npy preferred)")
    p.add_argument("--T_npz_key", default=None)
    p.add_argument(
        "--T_layout",
        choices=[
            "auto",
            "pixels_by_wavelength",
            "wavelength_by_pixels",
            "wavelength_last",
            "wavelength_first",
        ],
        default="auto",
    )
    p.add_argument(
        "--T_normalization",
        choices=["none", "column_l2"],
        default="none",
        help="Column L2 changes wavelength-dependent throughput; use only if experimentally intended",
    )
    p.add_argument(
        "--T_column_scale_path",
        default=None,
        help=(
            "Optional positive .npy multiplier per wavelength; e.g. save 1/P_laser(lambda) "
            "to remove calibration-source power drift"
        ),
    )
    p.add_argument("--wl_start_nm", type=float, default=1520.0)
    p.add_argument("--wl_end_nm", type=float, default=1580.0)
    p.add_argument(
        "--wl_step_nm",
        type=float,
        default=0.010,
        help="If supplied, strictly validates the T column count; e.g. 0.002 for 2 pm",
    )
    p.add_argument("--compress_dim", type=int, default=800)
    p.add_argument(
        "--projection_path",
        default=str(DEFAULT_PROJECTION_PATH),
        help="Reuse the original trained/tested C matrix for an apples-to-apples robust run",
    )
    p.add_argument("--projection_seed", type=int, default=2024)
    p.add_argument("--projection_storage_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument(
        "--physics_build_mode",
        choices=["auto", "full_gpu", "ram_stream", "disk_stream"],
        default="disk_stream",
        help=(
            "disk_stream reads T in wavelength blocks; auto/full_gpu/ram_stream are optional "
            "acceleration modes"
        ),
    )
    p.add_argument(
        "--physics_gpu_memory_fraction",
        type=float,
        default=0.82,
        help="Maximum fraction of currently free VRAM used by auto full-GPU selection",
    )
    p.add_argument("--physics_wavelength_block", type=int, default=128)
    p.add_argument("--force_rebuild_physics", action="store_true")

    # Universal dataset. Single-line is an explicit subset of the sparse mode.
    p.add_argument("--num_train_samples", type=int, default=200_000)
    p.add_argument("--num_val_samples", type=int, default=2_048)
    p.add_argument("--num_mode_val_samples", type=int, default=512)
    p.add_argument("--prob_single_line", type=float, default=0.30)
    p.add_argument("--prob_sparse", type=float, default=0.20)
    p.add_argument("--prob_doublet", type=float, default=0.25)
    p.add_argument("--prob_continuous", type=float, default=0.25)
    p.add_argument("--narrow_sigma_min_nm", type=float, default=0.003)
    p.add_argument("--narrow_sigma_max_nm", type=float, default=0.040)
    p.add_argument("--doublet_sigma_min_nm", type=float, default=0.003)
    p.add_argument("--doublet_sigma_max_nm", type=float, default=0.006)
    p.add_argument("--doublet_spacing_min_nm", type=float, default=0.010)
    p.add_argument("--doublet_spacing_max_nm", type=float, default=0.030)
    p.add_argument("--broad_sigma_min_nm", type=float, default=1.0)
    p.add_argument("--broad_sigma_max_nm", type=float, default=10.0)
    p.add_argument("--polynomial_background_probability", type=float, default=0.5)

    # Decoder.
    p.add_argument("--hidden_dim", type=int, default=1536)
    p.add_argument("--num_res_blocks", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.0)

    # Robust noise injection. Dark-frame temporal RMS was measured as 0.5098
    # raw counts, so the absolute range brackets that value.
    p.add_argument(
        "--noise_model",
        choices=["projected_pixel_awgn", "relative_encoder_awgn", "hybrid_awgn"],
        default="hybrid_awgn",
    )
    p.add_argument("--noise_std_min", type=float, default=0.25, help="Pixel-noise std in raw counts")
    p.add_argument("--noise_std_max", type=float, default=1.00, help="Pixel-noise std in raw counts")
    p.add_argument("--relative_noise_std_min", type=float, default=0.0)
    p.add_argument("--relative_noise_std_max", type=float, default=0.03)
    p.add_argument("--encoder_drift_std_max", type=float, default=0.02)
    p.add_argument("--gain_jitter", type=float, default=0.10)

    # Composite loss: alpha=1, beta=2, gamma=0.1 in Supporting Note S5.
    p.add_argument("--loss_alpha", type=float, default=1.0)
    p.add_argument("--loss_beta", type=float, default=2.0)
    p.add_argument("--loss_gamma", type=float, default=0.1)
    p.add_argument("--peak_weight", type=float, default=20.0)
    p.add_argument("--loss_emd_weight", type=float, default=0.25)
    p.add_argument("--consistency_weight", type=float, default=0.20)

    # Validation and model selection. Two bins correspond to 20 pm here, the
    # closest grid-representable tolerance to the experimentally verified 15 pm.
    p.add_argument("--selection_single_weight", type=float, default=1.0)
    p.add_argument("--selection_peak_penalty", type=float, default=1.0)
    p.add_argument("--ideal_scan_every", type=int, default=5)
    p.add_argument("--ideal_scan_batch_size", type=int, default=64)

    # Optimization. Batch 16 and lr=1e-3 match the final values stated in S5.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-6)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument(
        "--fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Runtime and outputs.
    p.add_argument("--out_dir", default=str(DEFAULT_ROBUST_RUN_DIR))
    p.add_argument("--resume", default=None, help="Path to training_latest.pt")
    p.add_argument("--device", default=None, help="Example: cuda:0 or cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_viz_samples", type=int, default=5)
    p.add_argument("--viz_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--deterministic", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
