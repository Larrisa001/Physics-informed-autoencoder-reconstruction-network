#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct relative spectra from experimental speckle images.

This script is the inference counterpart of
``train_physics_informed_spectrometer.py`` (format version 2):

    raw speckle - mean dark -> flatten -> C @ pixels -> decoder -> spectrum

Important
---------
* The transmission matrix used for training was already dark-subtracted.  The
  experimental speckle must therefore also be dark-subtracted exactly once.
* Do not normalize the camera image to [0, 1], divide by 255/65535, perform L2
  normalization, or standardize it.  Camera counts are retained.  The decoder
  itself contains LayerNorm and returns a peak-normalized relative spectrum.
* ROI, rotations/flips, and flatten order must be identical to those used when
  the transmission matrix was constructed.  A matching pixel count alone does
  not prove that the order is correct.

The program accepts one image, several images, or a directory.  It writes one
CSV and one plot per image, a combined NPZ file, a summary CSV, and the mean
dark image used for subtraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpg",
    ".jpeg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
}

# -----------------------------------------------------------------------------
# One-click defaults
# -----------------------------------------------------------------------------
# The script is expected to be placed directly under:
#   D:\mt\python\fiber_spectrometer\experiment3
# Paths are resolved from this file instead of from PyCharm's working directory,
# so clicking Run works even when the IDE uses a different working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "metafiber_1520-1580_robust"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "neural_network" / "1520-1580" / "test"
DEFAULT_BACKGROUND_DIR = SCRIPT_DIR / "neural_network" / "1520-1580"/"dark"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints" / "decoder_best.pt"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "experimental_reconstruction"


# -----------------------------------------------------------------------------
# Decoder: kept exactly consistent with the format-v2 training code
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))
        return x + residual


class SpectralDecoder(nn.Module):
    """Compressed measurement -> non-negative peak-normalized spectrum."""

    def __init__(
        self,
        compress_dim: int,
        spectrum_dim: int,
        hidden_dim: int = 1536,
        num_res_blocks: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(compress_dim)
        self.head = nn.Sequential(nn.Linear(compress_dim, hidden_dim), nn.GELU())
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, spectrum_dim))
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(z)
        x = self.head(x)
        x = self.body(x)
        spectrum = self.softplus(self.tail(x))
        peak = spectrum.amax(dim=1, keepdim=True).clamp_min(1e-8)
        return spectrum / peak


# -----------------------------------------------------------------------------
# File discovery and image preprocessing
# -----------------------------------------------------------------------------


def natural_key(path: Path) -> List[object]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def discover_images(path: Path, recursive: bool = False) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return files


def read_image(path: Path, color_mode: str) -> np.ndarray:
    """Read an image without rescaling its original numeric values."""
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as pil_image:
            if getattr(pil_image, "n_frames", 1) != 1:
                raise ValueError(f"Multi-frame image is not supported: {path}")
            image = np.asarray(pil_image)

    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 3:
        channels = image.shape[-1]
        if channels not in (3, 4):
            raise ValueError(f"Unsupported image shape {image.shape}: {path}")
        rgb = image[..., :3].astype(np.float32, copy=False)
        if color_mode == "reject":
            raise ValueError(
                f"Color image {image.shape} found at {path}. "
                "Use grayscale raw camera files or explicitly select --color_mode."
            )
        if color_mode == "channel0":
            image = rgb[..., 0]
        elif color_mode == "mean":
            image = rgb.mean(axis=-1)
        elif color_mode == "luma":
            image = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            raise ValueError(f"Unknown color mode: {color_mode}")
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got {image.shape}: {path}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Non-numeric image dtype {image.dtype}: {path}")

    output = np.asarray(image, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"Image contains NaN or Inf: {path}")
    return output


def transform_image(
    image: np.ndarray,
    roi: Optional[Sequence[int]],
    rotate: int,
    flip_x: bool,
    flip_y: bool,
    transpose: bool,
) -> np.ndarray:
    """Apply the same deterministic geometry to dark and signal images."""
    if roi is not None:
        x, y, width, height = [int(v) for v in roi]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid ROI {tuple(roi)}; expected X Y WIDTH HEIGHT")
        if x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError(f"ROI {tuple(roi)} exceeds image shape {image.shape}")
        image = image[y : y + height, x : x + width]

    if rotate:
        image = np.rot90(image, k=rotate // 90)
    if flip_x:
        image = np.fliplr(image)
    if flip_y:
        image = np.flipud(image)
    if transpose:
        image = image.T
    return np.ascontiguousarray(image, dtype=np.float32)


def load_and_transform(path: Path, args: argparse.Namespace) -> np.ndarray:
    return transform_image(
        read_image(path, args.color_mode),
        roi=args.roi,
        rotate=args.rotate,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        transpose=args.transpose,
    )


def build_mean_dark(
    dark_files: Sequence[Path], args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean dark and per-pixel temporal standard deviation."""
    first = load_and_transform(dark_files[0], args)
    mean = np.zeros_like(first, dtype=np.float64)
    m2 = np.zeros_like(first, dtype=np.float64)

    for count, path in enumerate(dark_files, start=1):
        frame = first if count == 1 else load_and_transform(path, args)
        if frame.shape != first.shape:
            raise ValueError(
                f"Dark image shape mismatch: {path} has {frame.shape}, expected {first.shape}"
            )
        delta = frame.astype(np.float64) - mean
        mean += delta / count
        m2 += delta * (frame.astype(np.float64) - mean)

    if len(dark_files) > 1:
        temporal_std = np.sqrt(m2 / (len(dark_files) - 1))
    else:
        temporal_std = np.zeros_like(mean)
    return mean.astype(np.float32), temporal_std.astype(np.float32)


def flatten_pixels(image: np.ndarray, order: str) -> np.ndarray:
    return np.asarray(image, dtype=np.float32).ravel(order=order).copy()


# -----------------------------------------------------------------------------
# Checkpoint and physics files
# -----------------------------------------------------------------------------


def select_device(device_arg: Optional[str]) -> torch.device:
    device = torch.device(device_arg or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device}) but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device.index)
    return device


def load_checkpoint(path: Path, device: torch.device) -> Tuple[SpectralDecoder, np.ndarray, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"decoder_state_dict", "decoder_config", "wavelength_nm"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")

    cfg = checkpoint["decoder_config"]
    for key in ("compress_dim", "spectrum_dim", "hidden_dim", "num_res_blocks", "dropout"):
        if key not in cfg:
            raise KeyError(f"decoder_config is missing {key!r}")
    decoder = SpectralDecoder(
        compress_dim=int(cfg["compress_dim"]),
        spectrum_dim=int(cfg["spectrum_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        dropout=float(cfg["dropout"]),
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    decoder.eval().to(device)

    wavelength_nm = np.asarray(checkpoint["wavelength_nm"], dtype=np.float64)
    if wavelength_nm.ndim != 1 or wavelength_nm.size != int(cfg["spectrum_dim"]):
        raise ValueError("Checkpoint wavelength axis is inconsistent with decoder_config")
    if not np.isfinite(wavelength_nm).all() or np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("Checkpoint wavelength axis is invalid")
    return decoder, wavelength_nm, checkpoint


def resolve_physics_path(
    explicit_path: Optional[str],
    checkpoint: Dict,
    checkpoint_path: Path,
    physics_key: str,
    fallback_name: str,
    required: bool,
) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    else:
        stored = checkpoint.get("physics_files", {}).get(physics_key)
        if stored:
            candidates.append(Path(stored))
        run_root = checkpoint_path.resolve().parent.parent
        candidates.append(run_root / "physics" / fallback_name)

    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return candidate
    if required:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find {physics_key}; tried: {tried}")
    return None


def load_projection(path: Path, expected_rows: int) -> Tuple[np.ndarray, int]:
    C_disk = np.load(path, mmap_mode="r", allow_pickle=False)
    if C_disk.ndim != 2:
        raise ValueError(f"Projection matrix must be 2-D, got {C_disk.shape}")
    if C_disk.shape[0] != expected_rows:
        raise ValueError(
            f"Projection rows {C_disk.shape[0]} != decoder compress_dim {expected_rows}"
        )
    if not np.issubdtype(C_disk.dtype, np.floating):
        raise TypeError(f"Projection dtype must be floating point, got {C_disk.dtype}")
    # A writable float32 copy prevents the read-only memmap PyTorch warning.
    C = np.array(C_disk, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(C).all():
        raise ValueError("Projection matrix contains NaN or Inf")
    return C, int(C.shape[1])


def load_effective_matrix(
    path: Optional[Path], expected_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    if path is None:
        return None
    W = np.load(path, mmap_mode="r", allow_pickle=False)
    if W.shape != expected_shape:
        print(
            f"Warning: ignoring W_eff with shape {W.shape}; expected {expected_shape}. "
            "Physics-consistency metrics will not be computed."
        )
        return None
    return W


def parse_true_wavelength_nm(path: Path) -> Optional[float]:
    """Extract labels such as 1481.000 from ``ccd_1481.000nm.bmp``."""
    matches = re.findall(r"(?<![\d.])(\d{3,4}(?:\.\d+)?)\s*nm", path.stem, flags=re.IGNORECASE)
    if not matches:
        return None
    return float(matches[-1])


def centered_column_norms(W_eff: np.ndarray, row_block: int = 64) -> np.ndarray:
    """L2 norms of W_eff columns after subtracting each column mean."""
    rows, columns = W_eff.shape
    sums = np.zeros(columns, dtype=np.float64)
    square_sums = np.zeros(columns, dtype=np.float64)
    for i0 in range(0, rows, row_block):
        block = np.asarray(W_eff[i0 : i0 + row_block], dtype=np.float64)
        sums += block.sum(axis=0)
        square_sums += np.einsum("ij,ij->j", block, block)
    centered_square_sums = square_sums - sums * sums / rows
    return np.sqrt(np.maximum(centered_square_sums, 1e-24))


def matched_filter_scan(
    z_measured: np.ndarray,
    W_eff: np.ndarray,
    centered_norms: np.ndarray,
    wavelength_nm: np.ndarray,
    true_wavelength_nm: Optional[float],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Pearson/cosine matching in the same feature-centering space as LayerNorm."""
    z64 = np.asarray(z_measured, dtype=np.float64)
    z_centered = z64 - z64.mean()
    z_norm = float(np.linalg.norm(z_centered))
    if z_norm <= 1e-20:
        raise ValueError("Compressed experimental measurement has zero centered norm")

    # Because sum(z_centered) == 0, z_centered @ W is exactly the dot product
    # with the mean-centered columns of W. Float32 matmul is sufficient here.
    dots = np.asarray(
        z_centered.astype(np.float32) @ W_eff,
        dtype=np.float64,
    )
    scores = dots / np.maximum(z_norm * centered_norms, 1e-24)
    scores = np.clip(scores, -1.0, 1.0)
    best_index = int(np.nanargmax(scores))

    top_count = min(3, scores.size)
    top_indices = np.argpartition(scores, -top_count)[-top_count:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    second_score = float(scores[top_indices[1]]) if top_count >= 2 else float("nan")

    result: Dict[str, float] = {
        "matched_peak_index": float(best_index),
        "matched_peak_nm": float(wavelength_nm[best_index]),
        "matched_best_score": float(scores[best_index]),
        "matched_second_score": second_score,
        "matched_score_margin": float(scores[best_index] - second_score),
        "matched_second_peak_nm": (
            float(wavelength_nm[top_indices[1]]) if top_count >= 2 else float("nan")
        ),
        "matched_third_peak_nm": (
            float(wavelength_nm[top_indices[2]]) if top_count >= 3 else float("nan")
        ),
    }
    if true_wavelength_nm is not None:
        true_index = int(np.argmin(np.abs(wavelength_nm - true_wavelength_nm)))
        result.update(
            {
                "true_grid_index": float(true_index),
                "true_grid_nm": float(wavelength_nm[true_index]),
                "true_column_centered_cosine": float(scores[true_index]),
                "matched_peak_error_nm": float(wavelength_nm[best_index] - true_wavelength_nm),
            }
        )
    return result, scores.astype(np.float32)


@torch.inference_mode()
def run_full_ideal_decoder_scan(
    decoder: SpectralDecoder,
    W_eff: np.ndarray,
    wavelength_nm: np.ndarray,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Feed every ideal monochromatic W_eff column through the decoder."""
    if batch_size <= 0:
        raise ValueError("ideal_scan_batch_size must be positive")
    predicted_nm = np.empty(wavelength_nm.size, dtype=np.float64)
    for i0 in tqdm(
        range(0, wavelength_nm.size, batch_size),
        desc="Ideal decoder scan",
        unit="batch",
    ):
        i1 = min(i0 + batch_size, wavelength_nm.size)
        z_block = np.array(W_eff[:, i0:i1].T, dtype=np.float32, order="C", copy=True)
        z_tensor = torch.from_numpy(z_block).to(device=device, dtype=torch.float32)
        reconstruction = decoder(z_tensor)
        peak_indices = reconstruction.argmax(dim=1).cpu().numpy()
        predicted_nm[i0:i1] = wavelength_nm[peak_indices]

    error_nm = predicted_nm - wavelength_nm
    abs_error_nm = np.abs(error_nm)
    metrics: Dict[str, float] = {
        "sample_count": float(wavelength_nm.size),
        "mae_nm": float(abs_error_nm.mean()),
        "median_absolute_error_nm": float(np.median(abs_error_nm)),
        "rmse_nm": float(np.sqrt(np.mean(error_nm * error_nm))),
        "max_absolute_error_nm": float(abs_error_nm.max()),
    }
    for tolerance_pm in (10, 20, 50, 100, 500, 1000):
        metrics[f"accuracy_within_{tolerance_pm}_pm"] = float(
            np.mean(abs_error_nm <= tolerance_pm / 1000.0 + 1e-12)
        )

    csv_path = output_dir / "ideal_decoder_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true_wavelength_nm", "predicted_peak_nm", "peak_error_nm"])
        writer.writerows(zip(wavelength_nm, predicted_nm, error_nm))
    with open(output_dir / "ideal_decoder_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(wavelength_nm, error_nm * 1000.0, color="#1f77b4", lw=0.8)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(50.0, color="#d62728", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(-50.0, color="#d62728", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlabel("True wavelength (nm)")
    ax.set_ylabel("Ideal-input peak error (pm)")
    ax.set_title(
        f"Full ideal decoder scan | MAE={metrics['mae_nm'] * 1000:.2f} pm | "
        f"within 50 pm={metrics['accuracy_within_50_pm'] * 100:.2f}%"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ideal_decoder_scan.png", dpi=200)
    plt.close(fig)
    return predicted_nm, metrics


# -----------------------------------------------------------------------------
# Reconstruction and outputs
# -----------------------------------------------------------------------------


def safe_stem(path: Path, used: Dict[str, int]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "speckle"
    number = used.get(stem, 0)
    used[stem] = number + 1
    return stem if number == 0 else f"{stem}_{number + 1}"


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-20 else float("nan")


def physics_consistency(
    z_measured: np.ndarray,
    spectrum: np.ndarray,
    W_eff: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """Cosine fidelity and gain-fitted NRMSE between z and W_eff @ spectrum."""
    if W_eff is None:
        return float("nan"), float("nan"), float("nan")
    z_forward = np.asarray(W_eff @ spectrum.astype(np.float32), dtype=np.float64)
    z = np.asarray(z_measured, dtype=np.float64)
    fidelity = cosine_similarity_np(z, z_forward)
    denom = float(np.dot(z_forward, z_forward))
    gain = float(np.dot(z, z_forward) / denom) if denom > 1e-20 else float("nan")
    z_norm = float(np.linalg.norm(z))
    nrmse = (
        float(np.linalg.norm(z - gain * z_forward) / z_norm)
        if z_norm > 1e-20 and math.isfinite(gain)
        else float("nan")
    )
    return fidelity, nrmse, gain


def diagnostic_label(
    network_error_nm: Optional[float],
    matched_error_nm: Optional[float],
    ideal_error_nm: Optional[float],
    tolerance_nm: float,
) -> str:
    if network_error_nm is None or matched_error_nm is None or ideal_error_nm is None:
        return "unknown_missing_true_wavelength_or_W_eff"
    network_pass = abs(network_error_nm) <= tolerance_nm
    matched_pass = abs(matched_error_nm) <= tolerance_nm
    ideal_pass = abs(ideal_error_nm) <= tolerance_nm
    if network_pass and matched_pass and ideal_pass:
        return "pass"
    if ideal_pass and matched_pass and not network_pass:
        return "experimental_noise_robustness_or_decoder_generalization_issue"
    if not ideal_pass and matched_pass:
        return "decoder_training_issue"
    if ideal_pass and not matched_pass:
        return "experimental_speckle_vs_T_mismatch"
    return "decoder_training_and_experimental_T_mismatch"


def save_plot(
    path: Path,
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    title: str,
    matched_scores: Optional[np.ndarray] = None,
    true_wavelength_nm: Optional[float] = None,
    matched_peak_nm: Optional[float] = None,
    ideal_decoder_peak_nm: Optional[float] = None,
) -> None:
    rows = 2 if matched_scores is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(10.5, 7.2 if rows == 2 else 4.6), sharex=True)
    axes_array = np.atleast_1d(axes)
    ax = axes_array[0]
    ax.plot(wavelength_nm, spectrum, color="#d62728", lw=1.2, label="Neural reconstruction")
    if true_wavelength_nm is not None:
        ax.axvline(true_wavelength_nm, color="black", lw=1.1, ls="--", label="True wavelength")
    if matched_peak_nm is not None:
        ax.axvline(matched_peak_nm, color="#1f77b4", lw=1.0, ls=":", label="Matched-filter peak")
    if ideal_decoder_peak_nm is not None:
        ax.axvline(
            ideal_decoder_peak_nm,
            color="#2ca02c",
            lw=1.0,
            ls="-.",
            label="Ideal-input decoder peak",
        )
    ax.set_ylabel("Relative intensity")
    ax.set_ylim(bottom=-0.02, top=max(1.04, float(spectrum.max()) * 1.04))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if true_wavelength_nm is not None or matched_peak_nm is not None:
        ax.legend(frameon=False, fontsize=8, ncol=2)

    if matched_scores is not None:
        match_ax = axes_array[1]
        match_ax.plot(wavelength_nm, matched_scores, color="#1f77b4", lw=0.8)
        if true_wavelength_nm is not None:
            match_ax.axvline(true_wavelength_nm, color="black", lw=1.0, ls="--")
        if matched_peak_nm is not None:
            match_ax.axvline(matched_peak_nm, color="#1f77b4", lw=1.0, ls=":")
        match_ax.set_ylabel("Centered cosine")
        match_ax.set_xlabel("Wavelength (nm)")
        match_ax.grid(alpha=0.25)
    else:
        ax.set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_spectrum_csv(path: Path, wavelength_nm: np.ndarray, spectrum: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_nm", "relative_intensity"])
        writer.writerows(zip(wavelength_nm, spectrum))


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    device = select_device(args.device)
    decoder, wavelength_nm, checkpoint = load_checkpoint(checkpoint_path, device)
    decoder_cfg = checkpoint["decoder_config"]
    compress_dim = int(decoder_cfg["compress_dim"])
    spectrum_dim = int(decoder_cfg["spectrum_dim"])

    projection_path = resolve_physics_path(
        args.projection,
        checkpoint,
        checkpoint_path,
        "projection_matrix",
        "projection_matrix.npy",
        required=True,
    )
    assert projection_path is not None
    C_np, n_pixels = load_projection(projection_path, compress_dim)
    C = torch.from_numpy(C_np).to(device=device, dtype=torch.float32)

    w_eff_path = resolve_physics_path(
        args.effective_matrix,
        checkpoint,
        checkpoint_path,
        "effective_matrix",
        "W_eff.npy",
        required=False,
    )
    W_eff = load_effective_matrix(w_eff_path, (compress_dim, spectrum_dim))

    background_dir = Path(args.background_dir)
    dark_files = discover_images(background_dir, recursive=args.recursive_background)
    if args.expected_background_count > 0 and len(dark_files) != args.expected_background_count:
        raise ValueError(
            f"Expected exactly {args.expected_background_count} background images, "
            f"found {len(dark_files)} in {background_dir}"
        )
    mean_dark, dark_temporal_std = build_mean_dark(dark_files, args)
    if mean_dark.size != n_pixels:
        raise ValueError(
            f"Processed background shape {mean_dark.shape} has {mean_dark.size} pixels, "
            f"but C expects {n_pixels}. Set --roi/geometry/flatten options to match T."
        )

    input_files: List[Path] = []
    for raw_input in args.input:
        input_files.extend(discover_images(Path(raw_input), recursive=args.recursive_input))
    # Preserve command-line grouping but remove accidental duplicates.
    input_files = list(dict.fromkeys(p.resolve() for p in input_files))
    if not input_files:
        raise FileNotFoundError("No experimental speckle images were selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mean_dark.npy", mean_dark)
    np.save(output_dir / "dark_temporal_std.npy", dark_temporal_std)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Projection: {projection_path}  shape={C_np.shape}")
    print(
        f"Wavelengths: {wavelength_nm[0]:.6f}-{wavelength_nm[-1]:.6f} nm, "
        f"samples={wavelength_nm.size}, "
        f"step={np.median(np.diff(wavelength_nm)) * 1000:.6f} pm"
    )
    print(
        f"Background: {len(dark_files)} images, processed shape={mean_dark.shape}, "
        f"mean={mean_dark.mean():.6g}, temporal RMS="
        f"{np.sqrt(np.mean(dark_temporal_std.astype(np.float64) ** 2)):.6g} counts"
    )
    print("Pixel values are kept in raw camera units; no [0,1]/L2/peak scaling is applied.")
    centered_w_norms: Optional[np.ndarray] = None
    ideal_predicted_nm: Optional[np.ndarray] = None
    ideal_scan_metrics: Optional[Dict[str, float]] = None
    if W_eff is None:
        print("Note: W_eff was not found; reconstruction works, but consistency metrics are disabled.")
    else:
        print("Preparing centered W_eff column norms for full-band matched filtering ...")
        centered_w_norms = centered_column_norms(W_eff)
        if not args.skip_full_ideal_scan:
            ideal_predicted_nm, ideal_scan_metrics = run_full_ideal_decoder_scan(
                decoder=decoder,
                W_eff=W_eff,
                wavelength_nm=wavelength_nm,
                device=device,
                batch_size=args.ideal_scan_batch_size,
                output_dir=output_dir,
            )
            print(
                "Ideal decoder scan: "
                f"MAE={ideal_scan_metrics['mae_nm'] * 1000:.3f} pm, "
                f"RMSE={ideal_scan_metrics['rmse_nm'] * 1000:.3f} pm, "
                f"within 50 pm={ideal_scan_metrics['accuracy_within_50_pm'] * 100:.2f}%"
            )

    used_stems: Dict[str, int] = {}
    spectra: List[np.ndarray] = []
    z_values: List[np.ndarray] = []
    matched_scores_all: List[np.ndarray] = []
    source_names: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    for index, image_path in enumerate(input_files, start=1):
        raw = load_and_transform(image_path, args)
        if raw.shape != mean_dark.shape:
            raise ValueError(
                f"Signal image shape mismatch: {image_path} has {raw.shape}, "
                f"background has {mean_dark.shape}"
            )
        saturation_fraction = float(
            np.mean(raw >= args.saturation_value - args.saturation_tolerance)
        )
        corrected = raw - mean_dark
        negative_fraction = float(np.mean(corrected < 0))
        if args.clip_negative:
            corrected = np.maximum(corrected, 0.0)
        pixels = flatten_pixels(corrected, args.flatten_order)
        if pixels.size != n_pixels:
            raise ValueError(
                f"Processed signal has {pixels.size} pixels but C expects {n_pixels}: {image_path}"
            )
        if float(np.linalg.norm(pixels)) <= 1e-12:
            raise ValueError(f"Dark-corrected signal is zero or nearly zero: {image_path}")

        pixel_tensor = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
        z_tensor = F.linear(pixel_tensor.unsqueeze(0), C)
        spectrum_tensor = decoder(z_tensor)
        z = z_tensor.squeeze(0).float().cpu().numpy()
        spectrum = spectrum_tensor.squeeze(0).float().cpu().numpy()
        spectrum /= max(float(spectrum.max()), 1e-12)

        fidelity, nrmse, fitted_gain = physics_consistency(z, spectrum, W_eff)
        peak_index = int(np.argmax(spectrum))
        peak_nm = float(wavelength_nm[peak_index])
        true_wavelength_nm = parse_true_wavelength_nm(image_path)
        network_error_nm = (
            peak_nm - true_wavelength_nm if true_wavelength_nm is not None else None
        )

        matched_result: Dict[str, float] = {}
        matched_scores: Optional[np.ndarray] = None
        if W_eff is not None and centered_w_norms is not None:
            matched_result, matched_scores = matched_filter_scan(
                z_measured=z,
                W_eff=W_eff,
                centered_norms=centered_w_norms,
                wavelength_nm=wavelength_nm,
                true_wavelength_nm=true_wavelength_nm,
            )
            matched_scores_all.append(matched_scores)

        ideal_decoder_peak_nm: Optional[float] = None
        ideal_decoder_error_nm: Optional[float] = None
        true_grid_index: Optional[int] = None
        if true_wavelength_nm is not None:
            true_grid_index = int(np.argmin(np.abs(wavelength_nm - true_wavelength_nm)))
            if ideal_predicted_nm is not None:
                ideal_decoder_peak_nm = float(ideal_predicted_nm[true_grid_index])
                ideal_decoder_error_nm = ideal_decoder_peak_nm - true_wavelength_nm

        matched_peak_nm = (
            float(matched_result["matched_peak_nm"]) if matched_result else None
        )
        matched_error_nm = (
            float(matched_result["matched_peak_error_nm"])
            if "matched_peak_error_nm" in matched_result
            else None
        )
        diagnosis = diagnostic_label(
            network_error_nm=network_error_nm,
            matched_error_nm=matched_error_nm,
            ideal_error_nm=ideal_decoder_error_nm,
            tolerance_nm=args.diagnostic_tolerance_nm,
        )

        stem = safe_stem(image_path, used_stems)
        write_spectrum_csv(output_dir / f"{stem}_spectrum.csv", wavelength_nm, spectrum)
        title = f"{image_path.name} | peak={peak_nm:.4f} nm"
        if math.isfinite(fidelity):
            title += f" | physics cosine={fidelity:.4f}"
        save_plot(
            output_dir / f"{stem}_spectrum.png",
            wavelength_nm,
            spectrum,
            title,
            matched_scores=matched_scores,
            true_wavelength_nm=true_wavelength_nm,
            matched_peak_nm=matched_peak_nm,
            ideal_decoder_peak_nm=ideal_decoder_peak_nm,
        )

        spectra.append(spectrum)
        z_values.append(z)
        source_names.append(str(image_path))
        summary_rows.append(
            {
                "source_image": str(image_path),
                "true_wavelength_nm_from_filename": true_wavelength_nm,
                "peak_wavelength_nm": peak_nm,
                "network_peak_error_nm": network_error_nm,
                "ideal_decoder_peak_nm": ideal_decoder_peak_nm,
                "ideal_decoder_peak_error_nm": ideal_decoder_error_nm,
                "matched_filter_peak_nm": matched_peak_nm,
                "matched_filter_peak_error_nm": matched_error_nm,
                "true_column_centered_cosine": matched_result.get(
                    "true_column_centered_cosine", float("nan")
                ),
                "matched_best_score": matched_result.get("matched_best_score", float("nan")),
                "matched_second_peak_nm": matched_result.get(
                    "matched_second_peak_nm", float("nan")
                ),
                "matched_second_score": matched_result.get(
                    "matched_second_score", float("nan")
                ),
                "matched_score_margin": matched_result.get(
                    "matched_score_margin", float("nan")
                ),
                "diagnostic_label": diagnosis,
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_mean": float(raw.mean()),
                "saturation_value": args.saturation_value,
                "saturated_pixel_fraction": saturation_fraction,
                "corrected_min": float(corrected.min()),
                "corrected_max": float(corrected.max()),
                "corrected_mean": float(corrected.mean()),
                "negative_fraction_before_optional_clip": negative_fraction,
                "physics_cosine_fidelity": fidelity,
                "physics_gain_fitted_nrmse": nrmse,
                "physics_fitted_gain": fitted_gain,
            }
        )
        metric_text = f", physics cosine={fidelity:.5f}, NRMSE={nrmse:.5f}" if math.isfinite(fidelity) else ""
        known_text = ""
        if true_wavelength_nm is not None:
            known_text = f", true={true_wavelength_nm:.6f} nm, NN error={network_error_nm:+.6f} nm"
        if matched_peak_nm is not None:
            known_text += f", match={matched_peak_nm:.6f} nm"
            if matched_error_nm is not None:
                known_text += f" ({matched_error_nm:+.6f} nm)"
        if ideal_decoder_peak_nm is not None:
            known_text += f", ideal-decoder={ideal_decoder_peak_nm:.6f} nm"
        print(
            f"[{index}/{len(input_files)}] {image_path.name}: peak={peak_nm:.6f} nm, "
            f"raw=[{raw.min():.6g}, {raw.max():.6g}], "
            f"sat={saturation_fraction * 100:.4f}%, "
            f"dark-corrected=[{corrected.min():.6g}, {corrected.max():.6g}]"
            f"{known_text}{metric_text}, diagnosis={diagnosis}"
        )

    matched_scores_array = (
        np.stack(matched_scores_all)
        if matched_scores_all
        else np.empty((len(input_files), 0), dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "reconstructed_spectra.npz",
        wavelength_nm=wavelength_nm,
        relative_spectra=np.stack(spectra),
        compressed_measurements=np.stack(z_values),
        matched_filter_centered_cosine=matched_scores_array,
        source_images=np.asarray(source_names, dtype=str),
    )
    summary_path = output_dir / "reconstruction_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    def finite_values(key: str) -> np.ndarray:
        values = []
        for row in summary_rows:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    diagnostic_summary: Dict[str, object] = {
        "diagnostic_tolerance_nm": args.diagnostic_tolerance_nm,
        "labeled_experimental_image_count": int(
            sum(row["true_wavelength_nm_from_filename"] is not None for row in summary_rows)
        ),
        "diagnosis_counts": {},
        "full_ideal_decoder_scan": ideal_scan_metrics,
    }
    for row in summary_rows:
        label = str(row["diagnostic_label"])
        counts = diagnostic_summary["diagnosis_counts"]
        assert isinstance(counts, dict)
        counts[label] = int(counts.get(label, 0)) + 1

    for prefix, key in (
        ("experimental_neural_network", "network_peak_error_nm"),
        ("experimental_matched_filter", "matched_filter_peak_error_nm"),
        ("ideal_input_decoder", "ideal_decoder_peak_error_nm"),
    ):
        errors = finite_values(key)
        if errors.size:
            diagnostic_summary[prefix] = {
                "sample_count": int(errors.size),
                "mae_nm": float(np.mean(np.abs(errors))),
                "median_absolute_error_nm": float(np.median(np.abs(errors))),
                "rmse_nm": float(np.sqrt(np.mean(errors * errors))),
                "max_absolute_error_nm": float(np.max(np.abs(errors))),
                "accuracy_within_diagnostic_tolerance": float(
                    np.mean(np.abs(errors) <= args.diagnostic_tolerance_nm + 1e-12)
                ),
            }
    saturation_values = finite_values("saturated_pixel_fraction")
    diagnostic_summary["saturation"] = {
        "threshold_value": args.saturation_value,
        "mean_saturated_pixel_fraction": float(saturation_values.mean()),
        "max_saturated_pixel_fraction": float(saturation_values.max()),
    }
    with open(output_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2, ensure_ascii=False)

    settings = {
        "checkpoint": str(checkpoint_path.resolve()),
        "projection": str(projection_path.resolve()),
        "effective_matrix": str(w_eff_path.resolve()) if w_eff_path is not None else None,
        "background_dir": str(background_dir.resolve()),
        "background_count": len(dark_files),
        "processed_image_shape": list(mean_dark.shape),
        "pixel_count": n_pixels,
        "roi_xywh": list(args.roi) if args.roi is not None else None,
        "rotate_degrees_counterclockwise": args.rotate,
        "flip_x": args.flip_x,
        "flip_y": args.flip_y,
        "transpose": args.transpose,
        "flatten_order": args.flatten_order,
        "clip_negative_after_dark_subtraction": args.clip_negative,
        "color_mode": args.color_mode,
        "saturation_value": args.saturation_value,
        "saturation_tolerance": args.saturation_tolerance,
        "diagnostic_tolerance_nm": args.diagnostic_tolerance_nm,
        "full_ideal_scan_enabled": not args.skip_full_ideal_scan,
        "ideal_scan_batch_size": args.ideal_scan_batch_size,
        "output_normalization": "per-spectrum peak (L-infinity)",
    }
    with open(output_dir / "inference_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    print("Diagnostic counts:", diagnostic_summary["diagnosis_counts"])
    print(f"Finished. Results: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Experimental speckle -> dark subtraction -> C -> decoder -> relative spectrum",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT_DIR)],
        help="One or more speckle image files/directories",
    )
    parser.add_argument(
        "--background_dir",
        default=str(DEFAULT_BACKGROUND_DIR),
        help="Directory containing the dark/background frames",
    )
    parser.add_argument(
        "--expected_background_count",
        type=int,
        default=10,
        help="Require this many dark frames; use 0 to disable the count check",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to checkpoints/decoder_best.pt",
    )
    parser.add_argument(
        "--projection",
        default=None,
        help="Path to projection_matrix.npy; auto-resolved from checkpoint/run if omitted",
    )
    parser.add_argument(
        "--effective_matrix",
        default=None,
        help="Optional W_eff.npy for physics-consistency diagnostics",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu")

    diagnostics = parser.add_argument_group("physics diagnostics")
    diagnostics.add_argument(
        "--diagnostic_tolerance_nm",
        type=float,
        default=0.05,
        help="Peak-error tolerance used only for pass/fail diagnostic labels",
    )
    diagnostics.add_argument(
        "--ideal_scan_batch_size",
        type=int,
        default=64,
        help="GPU batch size for the full 16001-column ideal decoder scan",
    )
    diagnostics.add_argument(
        "--skip_full_ideal_scan",
        action="store_true",
        help="Skip the ideal W_eff-column decoder sweep (not recommended for diagnosis)",
    )
    diagnostics.add_argument(
        "--saturation_value",
        type=float,
        default=255.0,
        help="Raw pixel value treated as camera/file saturation",
    )
    diagnostics.add_argument(
        "--saturation_tolerance",
        type=float,
        default=1e-6,
        help="Tolerance below saturation_value used in the saturation count",
    )

    geometry = parser.add_argument_group("camera geometry (must match T construction)")
    geometry.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    geometry.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    geometry.add_argument("--flip_x", action="store_true", help="Left-right flip after crop/rotation")
    geometry.add_argument("--flip_y", action="store_true", help="Top-bottom flip after crop/rotation")
    geometry.add_argument("--transpose", action="store_true", help="Transpose after crop/rotation/flips")
    geometry.add_argument(
        "--flatten_order",
        choices=["C", "F"],
        default="C",
        help="NumPy row-major (C) or column-major (F) pixel order",
    )

    preprocessing = parser.add_argument_group("image preprocessing")
    preprocessing.add_argument(
        "--clip_negative",
        action="store_true",
        help="Clip negative values after dark subtraction; enable only if T construction did so",
    )
    preprocessing.add_argument(
        "--color_mode",
        choices=["reject", "channel0", "mean", "luma"],
        default="reject",
        help="How to handle color files; raw grayscale is strongly recommended",
    )
    preprocessing.add_argument("--recursive_input", action="store_true")
    preprocessing.add_argument("--recursive_background", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
