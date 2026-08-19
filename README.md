# Physics-Informed Spectral Reconstruction for a Speckle Spectrometer

This repository contains the training code for a physics-informed neural
network that reconstructs relative input spectra from camera-recorded speckle
patterns. The physical encoder is fixed by the experimentally calibrated
transmission matrix, while only the deep residual decoder is optimized.

The default configuration covers **1520–1580 nm** with a wavelength sampling
interval of **0.01 nm (10 pm)**, giving **6001 spectral sampling points**.

> **Scope.** This repository trains the fixed-physics spectral decoder. It does
> not train a learned optical encoder or an experimental-domain adapter.

## 1. Physical model

For a discretized incident spectrum $S$, the camera measurement is modeled as

$$
I = TS + \eta,
$$

where:

- $T\in\mathbb{R}^{P\times M}$ is the experimentally measured transmission
  matrix;
- $P$ is the number of camera pixels;
- $M$ is the number of wavelength samples;
- $I\in\mathbb{R}^{P}$ is the speckle-intensity vector; and
- $\eta$ represents measurement noise.

To reduce the input dimension, a fixed Gaussian projection matrix
$C\in\mathbb{R}^{K\times P}$ is applied:

$$
z = CI = (CT)S + C\eta.
$$

The effective physical encoder is therefore

$$
W_{\mathrm{eff}} = CT,
$$

and training uses

$$
z = W_{\mathrm{eff}}S.
$$

Only the decoder parameters are trainable. Both $C$ and
$W_{\mathrm{eff}}$ remain fixed throughout training.

## 2. Main features

- Memory-mapped loading of large `.npy` transmission matrices.
- Automatic handling of `[pixels, wavelengths]`, `[wavelengths, pixels]`,
  wavelength-first, and wavelength-last layouts.
- Block-streamed construction of $W_{\mathrm{eff}}=CT$, avoiding full
  loading of the transmission matrix into GPU memory.
- Reusable Gaussian projection matrix with row-wise L2 normalization.
- Deterministic generation of 200,000 synthetic training spectra.
- Four spectral modes: single-line, sparse, doublet, and continuous spectra.
- Hybrid noise injection in the compressed-measurement domain.
- ResNet decoder with non-negative, peak-normalized output.
- Composite spectral loss and two-view noise-consistency regularization.
- Clean, noisy, mode-specific, and full-band single-line validation.
- Mixed-precision training, fused AdamW when available, learning-rate decay,
  checkpointing, resumption, and early stopping.

## 3. Network architecture

The default decoder consists of:

1. Layer normalization of the 800-dimensional compressed measurement;
2. a fully connected projection head with 1536 hidden neurons;
3. ten pre-normalized residual MLP blocks;
4. a fully connected spectral output layer;
5. a Softplus activation for non-negative intensity; and
6. per-spectrum peak normalization.

The decoder output is a **relative spectrum** whose maximum value is normalized
to one. The code does not recover absolute optical power.

## 4. Synthetic training spectra

The default universal dataset contains four spectral categories:

| Mode | Default proportion | Description |
|---|---:|---|
| Single line | 0.30 | An exact one-hot line on the calibrated wavelength grid |
| Sparse | 0.20 | A superposition of 1–5 narrow Gaussian peaks |
| Doublet | 0.25 | Two narrow peaks separated by 10–30 pm |
| Continuous | 0.25 | A superposition of 3–10 broadband Gaussian envelopes, optionally with a polynomial background |

Every generated spectrum is constrained to be non-negative and is normalized
by its maximum value before entering the fixed physical encoder.

With the default 10 pm wavelength grid, separations are represented in integer
grid bins. For example, 10, 20, and 30 pm correspond to one, two, and three
sampling intervals, respectively.

## 5. Robust measurement model

The default `hybrid_awgn` model combines:

- projected pixel-level Gaussian noise;
- signal-relative Gaussian noise in the compressed domain;
- slowly varying encoder/background drift; and
- multiplicative gain jitter.

The absolute pixel-noise standard deviation must use the **same camera units as
the transmission matrix**. With the provided defaults, the pixel-noise standard
deviation is uniformly sampled from 0.25 to 1.00 raw counts.

The projected pixel-noise model uses the diagonal approximation

$$
\operatorname{Std}[(C\eta)_i]
= \sigma_{\mathrm{pixel}}\lVert C_i\rVert_2.
$$

Two independent noisy views of the same target spectrum are reconstructed in
each training step. A consistency term encourages the two reconstructions to
agree.

## 6. Loss function

The supervised reconstruction loss is

$$
\mathcal{L}_{\mathrm{sup}}
= \alpha\mathcal{L}_{\cos}
+ \beta\mathcal{L}_{\mathrm{amp}}
+ \gamma\mathcal{L}_{\mathrm{grad}}
+ w_{\mathrm{EMD}}\mathcal{L}_{\mathrm{EMD}},
$$

with the default weights

$$
\alpha=1,\qquad \beta=2,\qquad \gamma=0.1,
\qquad w_{\mathrm{EMD}}=0.25.
$$

The terms are:

- **Cosine loss:** spectral-shape similarity;
- **weighted L1 loss:** absolute-amplitude agreement, using
  $1+20S$ as the wavelength-dependent weight;
- **gradient loss:** agreement between adjacent spectral differences; and
- **EMD loss:** a one-dimensional cumulative-distribution distance that
  penalizes large wavelength displacements more strongly than local shifts.

The final objective is

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{sup}}
+ w_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}},
$$

where the default consistency weight is $w_{\mathrm{cons}}=0.20$.

## 7. Repository structure

Place the training script directly in the repository root and organize the
files as follows:

```text
.
├── train_physics_informed_spectrometer.py
├── README.md
├── neural_network/
│   └── 1520-1580/
│       └── T_ori.npy
└── runs/
    ├── metafiber_1520-1580/
    │   └── physics/
    │       └── projection_matrix.npy       # optional reusable C
    └── metafiber_1520-1580_robust/         # generated automatically
```

If `projection_matrix.npy` does not exist, the script creates it automatically.
Once a projection matrix has been used to train a decoder, retain it: inference
must use the identical matrix.

## 8. Transmission-matrix requirements

The preferred format is an uncompressed `.npy` file because it supports memory
mapping. `.npz` files are accepted but may load the complete matrix into RAM.

For a 512 × 640 camera image and the default wavelength grid, the expected
matrix shape is typically

```text
(327680, 6001)  # [pixels, wavelengths]
```

The script also supports transposed and three-dimensional layouts through
`--T_layout`.

### Background correction

The training script assumes that the transmission matrix has already been
background/dark corrected during calibration. It does not read dark frames.
Experimental inference images must therefore be corrected in the same way,
normally by subtracting the mean dark frame exactly once.

### Intensity scaling

Do not independently rescale experimental speckle images to `[0,1]` if the
transmission matrix is stored in raw camera counts. Training and inference must
use consistent units.

The default setting is

```text
--T_normalization none
```

Column-wise L2 normalization changes the wavelength-dependent throughput of
the calibrated system and should be enabled only when that change is physically
intended.

## 9. Installation

The code requires Python 3.9 or later. A CUDA-enabled PyTorch installation is
recommended for training.

Core dependencies:

```text
numpy
matplotlib
torch
tqdm
```

Create and activate a clean environment, then install the dependencies. For
example:

```bash
conda create -n speckle-spectrometer python=3.9 -y
conda activate speckle-spectrometer
pip install numpy matplotlib tqdm
# Install a CUDA-enabled PyTorch build appropriate for the local CUDA setup.
```

The reported implementation environment can be recorded automatically in
`environment.json` at the start of each run.

## 10. Quick start

The default paths are resolved relative to the script location, not the current
IDE working directory. If the files follow the structure above, start training
with:

```bash
python train_physics_informed_spectrometer.py
```

An equivalent explicit command is:

```bash
python train_physics_informed_spectrometer.py \
  --T_path neural_network/1520-1580/T_ori.npy \
  --T_layout pixels_by_wavelength \
  --wl_start_nm 1520 \
  --wl_end_nm 1580 \
  --wl_step_nm 0.010 \
  --compress_dim 800 \
  --hidden_dim 1536 \
  --num_res_blocks 10 \
  --batch_size 16 \
  --epochs 200 \
  --amp bf16 \
  --out_dir runs/metafiber_1520-1580_robust
```

On Windows PowerShell, the same command can be written on one line or use a
backtick instead of `\` for line continuation.

## 11. Default configuration

| Parameter | Default |
|---|---:|
| Wavelength range | 1520–1580 nm |
| Wavelength interval | 0.010 nm (10 pm) |
| Spectral samples | 6001 |
| Compressed dimension | 800 |
| Hidden dimension | 1536 |
| Residual blocks | 10 |
| Training spectra | 200,000 |
| Mixed validation spectra | 2,048 |
| Per-mode validation spectra | 512 |
| Batch size | 16 |
| Maximum epochs | 200 |
| Initial learning rate | $10^{-3}$ |
| Weight decay | $10^{-5}$ |
| LR reduction factor | 0.5 |
| LR patience | 5 epochs |
| Early-stopping patience | 30 epochs |
| Early-stopping minimum improvement | $10^{-6}$ |
| Mixed precision | BF16 |
| Physics build mode | Disk streaming |
| Wavelength block size | 128 |

## 12. Building the effective matrix

The default `disk_stream` mode reads the transmission matrix in wavelength
blocks and performs

```text
W_eff[:, j0:j1] = C @ T[:, j0:j1]
```

on the selected device. This mode minimizes RAM and VRAM use and is appropriate
for very large transmission matrices.

Available modes are:

| Mode | Description |
|---|---|
| `disk_stream` | Read wavelength blocks from the memory-mapped file |
| `ram_stream` | Load the complete transmission matrix into system RAM, then stream blocks to the GPU |
| `full_gpu` | Load the complete matrix into GPU memory and perform one matrix multiplication |
| `auto` | Select between full-GPU and disk-streaming modes from available VRAM |

The generated matrix and metadata are cached as

```text
runs/metafiber_1520-1580_robust/physics/W_eff.npy
runs/metafiber_1520-1580_robust/physics/W_eff_metadata.json
```

The cache is reused only when the transmission matrix, projection matrix, and
relevant physical settings match the recorded metadata.

To rebuild it deliberately, use:

```bash
python train_physics_informed_spectrometer.py --force_rebuild_physics
```

## 13. Checkpoints and resuming training

The script writes two kinds of checkpoints:

- `training_latest.pt`: complete state for exact training resumption, including
  optimizer, scheduler, random-number-generator, and mixed-precision states;
- `decoder_best.pt`: compact inference checkpoint selected by validation
  performance;
- `decoder_latest.pt`: compact decoder checkpoint from the most recent epoch.

Resume an interrupted run with:

```bash
python train_physics_informed_spectrometer.py \
  --resume runs/metafiber_1520-1580_robust/checkpoints/training_latest.pt
```

If a `training_latest.pt` file already exists and `--resume` is not supplied,
the program stops instead of silently overwriting the previous run. To start an
independent experiment, choose a new output directory:

```bash
python train_physics_informed_spectrometer.py \
  --out_dir runs/metafiber_1520-1580_robust_repeat01
```

## 14. Learning-rate scheduling and early stopping

The learning rate is reduced when the model-selection score stops improving.
The score is

$$
L_{\mathrm{select}}
=L_{\mathrm{mixed,noisy}}
+w_{\mathrm{single}}L_{\mathrm{single,noisy}}
+w_{\mathrm{peak}}(1-A_{\leq2\,\mathrm{bins}}).
$$

For the default 10 pm grid, two bins correspond to 20 pm.

Training stops when this score has not improved by at least $10^{-6}$ for 30
consecutive epochs. To disable early stopping while retaining the 200-epoch
maximum, use:

```bash
python train_physics_informed_spectrometer.py --early_stop_patience 0
```

The best checkpoint is retained even when early stopping is triggered.

## 15. Validation

The code evaluates:

- mixed clean spectra;
- mixed noisy spectra;
- clean single-line spectra;
- noisy single-line spectra;
- noisy sparse spectra;
- noisy doublet spectra;
- noisy continuous spectra; and
- a full ideal single-line scan over every calibrated wavelength column.

Important single-line metrics include:

- peak MAE and RMSE;
- accuracy within one bin (10 pm);
- accuracy within two bins (20 pm);
- accuracy within five bins (50 pm); and
- the number of catastrophic errors larger than 1 nm in the full ideal scan.

The full ideal scan evaluates the decoder on columns of $W_{\mathrm{eff}}$.
It verifies the learned inverse of the calibrated forward model, but it is not
a substitute for validation using independently acquired experimental speckle
images.

## 16. Outputs

A default run produces:

```text
runs/metafiber_1520-1580_robust/
├── checkpoints/
│   ├── decoder_best.pt
│   ├── decoder_latest.pt
│   └── training_latest.pt
├── figures/
│   ├── ideal_scan_epoch_XXX.json
│   ├── ideal_scan_epoch_XXX.png
│   └── validation_epoch_XXX.png
├── physics/
│   ├── projection_matrix.npy       # if generated inside this run
│   ├── W_eff.npy
│   ├── W_eff_metadata.json
│   └── wavelength_nm.npy
├── environment.json
├── history.csv
└── run_manifest.json
```

The decoder used for inference is:

```text
runs/metafiber_1520-1580_robust/checkpoints/decoder_best.pt
```

Experimental inference must use the same projection matrix and wavelength axis
recorded in the checkpoint and run manifest.

## 17. Reproducibility

The synthetic dataset is index-deterministic. A sample is determined by its
dataset seed and sample index, avoiding duplicated random sequences across
multiple DataLoader workers.

The training checkpoint stores Python, NumPy, PyTorch, CUDA, and DataLoader
random states. For stricter deterministic execution, use:

```bash
python train_physics_informed_spectrometer.py --deterministic
```

Strict determinism can reduce training speed and may change which optimized
GPU kernels are available.

## 18. Large files and GitHub

The transmission matrix and generated physical matrices can be many gigabytes
and should not be committed to ordinary Git history. A recommended `.gitignore`
is:

```gitignore
__pycache__/
*.py[cod]
.idea/
.vscode/

neural_network/**/T_ori.npy
runs/**/physics/W_eff.npy
runs/**/physics/projection_matrix.npy
runs/**/checkpoints/training_latest.pt
```

For reproducibility, provide the large calibration data through an institutional
repository, Zenodo, Figshare, or another archival data service, and add the
download URL and a SHA-256 checksum here:

```text
Transmission matrix: [DATA_URL]
SHA-256: [SHA256_CHECKSUM]
```

Small inference checkpoints may be distributed through a release or Git LFS,
subject to the repository's file-size policy.

## 19. Citation

If this code or dataset is useful in your research, please cite the associated
paper:

```bibtex
@article{AUTHOR_YEAR_SPECKLE_SPECTROMETER,
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  volume  = {[VOLUME]},
  pages   = {[PAGES]},
  doi     = {[DOI]}
}
```

Replace the bracketed fields after the paper metadata is finalized.

## 20. License

Add a `LICENSE` file before public release. A permissive license such as MIT or
BSD-3-Clause is commonly used for academic research code. If the calibration
data have different redistribution terms, document them separately.

## 21. Contact

For questions about the implementation, calibration data, or reproduction of
the reported results, please contact:

```text
[CORRESPONDING AUTHOR]
[INSTITUTION]
[EMAIL]
```
