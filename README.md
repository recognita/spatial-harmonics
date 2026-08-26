# spatial-harmonics

Spherical-harmonics-based neural layer that improves the 3D spatial-attention block used in speech representation models.

Presented at the Student Spring University Forum 2025.

## Overview

The baseline model's Spatial Attention layer maps EEG/MEG sensor geometry into the network using a fixed 2D projection. This project replaces it with a **3D Spatial Attention Matrix built from vector spherical harmonics**, giving the model a more accurate geometric interpretation of sensor positions while using fewer parameters.

## Method

Three model variants are compared:

1. **Baseline** — the original Défossez et al. spatial-attention formulation (`--variant baseline`).
2. **+ Spatial harmonics** — 3D spatial attention via vector spherical harmonics (`--variant harmonics_base`).
3. **+ Reduced parameters** — harmonics combined with a smaller Subject Block and an empirically-chosen optimum of **4 temporal convolutions** in the Subject Layer (`--variant harmonics_270`, the configuration behind the reported 40% reduction).

## Results

- **~40% reduction** in model parameter count.
- No loss in speech-representation accuracy versus the baseline — in some configurations, improved accuracy from the more geometrically-accurate spatial attention.

## Repository contents

| File | Purpose |
|---|---|
| `src/common.py` | Shared model components: loss functions, conv blocks, `BrainModule`, dataset, trainer |
| `src/spatial_attention.py` | The two spatial-attention layers: `FourierSpatialAttention` (baseline) and `HarmonicsSpatialAttention` (this project's contribution) |
| `src/train.py` | Training entry point — reproduces all three reported configurations via `--variant` |

This used to be three separate, ~95%-duplicated notebooks (`baseline270.ipynb`, `br6_sp3d_base.ipynb`, `br6_sp3d_270.ipynb`); they're now one shared codebase parameterized by `--variant`, which is both easier to read and removes the risk of the three copies silently drifting apart.

## Setup

```bash
pip install -r requirements.txt
python src/train.py --variant harmonics_270 --data-dir /path/to/MASC-MEG/process_v2/
```

## Tech stack

Python, PyTorch, NumPy, einops, SciPy, MNE
