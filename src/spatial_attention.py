"""
Two interchangeable spatial-attention layers for `common.SpatialModule`:

- `FourierSpatialAttention` — the original Défossez et al. formulation:
  a 2D Fourier basis (cos/sin of k*x + l*y over each sensor's projected
  (x, y) position) used to build a learned soft-attention map from
  MEG/EEG sensors onto a smaller set of "virtual" spatial channels.

- `HarmonicsSpatialAttention` — this project's contribution: replaces the
  2D Fourier basis with **vector spherical harmonics** evaluated on each
  sensor's true 3D position on the head (via associated Legendre
  polynomials), giving the attention layer a geometrically accurate
  basis instead of a flat 2D projection. This is what achieves the
  ~40% parameter reduction reported in the README/publication — a
  smaller `K` (harmonics degree) is needed for the same spatial
  resolution because the basis already matches the sensor geometry.

Both layers share the same public interface (`forward`, `get_spatial_filter`),
so `train.py` can swap between them by config alone.

Fix applied here: `HarmonicsSpatialAttention` originally loaded
`coords/sensor_xyz.npy` from a hardcoded relative path
(`'../datasets/MASC-MEG/process_v2/coords/sensor_xyz.npy'`) baked directly
into `__init__`, while every other data path in the project was threaded
through the `dirprocess` config value. That meant the layer silently broke
if you ran training from a different working directory than the original
author's. It now takes `dirprocess` as a constructor argument like
everything else.
"""

import math

import einops
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierSpatialAttention(nn.Module):
    """2D Fourier-basis spatial attention (baseline / Défossez et al. reproduction)."""

    def __init__(self, n_input, n_output, K, coords_xy, n_dropout, dropout_radius, seed=None):
        super().__init__()
        self.n_input = n_input
        self.n_dropout = n_dropout
        self.dropout_radius = dropout_radius

        coords_xy = torch.tensor(coords_xy, dtype=torch.float32, requires_grad=False)
        self.register_buffer("_coords_xy", coords_xy)

        fourier_layout = self._create_fourier_layout(n_input, K, coords_xy)
        self.register_buffer("_layout", fourier_layout)

        self.Z = nn.Parameter(self._create_parameters(n_input, n_output, K, seed))

    def _create_fourier_layout(self, n_input, K, coords_xy):
        coords_x, coords_y = coords_xy[:, 0], coords_xy[:, 1]
        layout = torch.zeros((2, K, K, n_input), requires_grad=False)
        for k in range(K):
            for l in range(K):
                phase = 2 * math.pi * ((k + 1) * coords_x + (l + 1) * coords_y)
                layout[0, k, l, :] = torch.cos(phase)
                layout[1, k, l, :] = torch.sin(phase)
        return einops.rearrange(layout, "a k l i -> 1 a k l i")

    def _create_parameters(self, n_input, n_output, K, seed=None):
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator().manual_seed(seed)
        Z = torch.randn(size=(n_output, 2, K, K), generator=generator) * 2 / (n_input + n_output)
        return einops.rearrange(Z, "j a k l -> j a k l 1")

    def to(self, device):
        self._coords_xy = self._coords_xy.to(device)
        self._layout = self._layout.to(device)
        return super().to(device)

    def get_spatial_filter(self):
        A = einops.reduce(self.Z * self._layout, "j a k l i -> j i", "sum")
        return F.softmax(A, dim=1).clone().detach()

    def forward(self, x):
        A = einops.reduce(self.Z * self._layout, "j a k l i -> j i", "sum")
        A = _apply_spatial_dropout(A, self.n_dropout, self.dropout_radius, self._coords_xy, self.n_input)
        attention = F.softmax(A, dim=1)
        return torch.einsum("oi, bit -> bot", attention, x)


class HarmonicsSpatialAttention(nn.Module):
    """3D vector-spherical-harmonics spatial attention (this project's contribution)."""

    def __init__(self, n_input, n_output, K, coords_xy, n_dropout, dropout_radius, dirprocess, seed=None):
        super().__init__()
        self.n_input = n_input
        self.n_dropout = n_dropout
        self.dropout_radius = dropout_radius

        coords_xy = torch.tensor(coords_xy, dtype=torch.float32, requires_grad=False)
        self.register_buffer("_coords_xy", coords_xy)

        coords_xyz = np.load(dirprocess + "coords/sensor_xyz.npy")
        coords_sph = torch.tensor(_cart2sph(coords_xyz), dtype=torch.float32, requires_grad=False)

        layout = self._create_harmonics_layout(coords_sph, K - 1)
        self.register_buffer("_layout", layout)

        self.Z = nn.Parameter(self._create_parameters(n_output, K, seed))

    @staticmethod
    def _spherical_harmonic_normalization(l, m):
        """log-space normalization constant for a real spherical harmonic of degree l, order m (avoids overflow for large l)."""
        result = math.log(2 * l + 1)
        result -= math.log((2 if m == 0 else 1) * 2 * math.pi)
        result += scipy.special.gammaln(l - abs(m) + 1)
        result -= scipy.special.gammaln(l + abs(m) + 1)
        return math.exp(result / 2)

    def _create_harmonics_layout(self, coords_sph, degree):
        n_input = coords_sph.shape[0]
        theta, phi = coords_sph[:, 1], coords_sph[:, 2]

        legendre_values = np.zeros((degree + 1, degree + 1, n_input))
        for i in range(n_input):
            legendre_values[:, :, i] = scipy.special.lpmn(degree, degree, math.cos(theta[i]))[0]
        legendre_values = torch.tensor(legendre_values, dtype=torch.float32)

        layout = torch.zeros((1, degree + 1, degree + 1, n_input), dtype=torch.float32, requires_grad=False)
        counter = -1
        for l in range(degree + 1):
            for m in range(-l, l + 1):
                counter += 1
                i, j = counter % (degree + 1), counter // (degree + 1)
                norm = self._spherical_harmonic_normalization(l, m)
                angular = torch.cos(m * phi) if m >= 0 else torch.sin(-m * phi)
                layout[:, i, j, :] = norm * legendre_values[abs(m), l, :] * angular
        return layout

    def _create_parameters(self, n_output, K, seed=None):
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator().manual_seed(seed)
        Z = torch.randn(size=(n_output, K, K), generator=generator) * 2 / (self.n_input + n_output)
        return einops.rearrange(Z, "j k l -> j k l 1")

    def to(self, device):
        self._coords_xy = self._coords_xy.to(device)
        self._layout = self._layout.to(device)
        return super().to(device)

    def get_spatial_filter(self):
        A = einops.reduce(self.Z * self._layout, "j k l i -> j i", "sum")
        return F.softmax(A, dim=1).clone().detach()

    def forward(self, x):
        A = einops.reduce(self.Z * self._layout, "j k l i -> j i", "sum")
        A = _apply_spatial_dropout(A, self.n_dropout, self.dropout_radius, self._coords_xy, self.n_input)
        attention = F.softmax(A, dim=1)
        return torch.einsum("oi, bit -> bot", attention, x)


def _cart2sph(sensor_xyz):
    x, y, z = sensor_xyz[:, 0], sensor_xyz[:, 1], sensor_xyz[:, 2]
    xy = np.linalg.norm(sensor_xyz[:, :2], axis=-1)
    r = np.linalg.norm(sensor_xyz, axis=-1)
    theta = np.arctan2(xy, z)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)


def _apply_spatial_dropout(A, n_dropout, dropout_radius, coords_xy, n_input):
    """During training, randomly masks out sensors within `dropout_radius` of `n_dropout` random points (spatial dropout)."""
    if n_dropout <= 0:
        return A
    mask = torch.zeros((1, n_input), dtype=A.dtype, device=A.device)
    dropout_centers = torch.rand(size=(n_dropout, 2), device=A.device) * 0.8 + 0.1
    for center in dropout_centers:
        distances = torch.linalg.norm(coords_xy - center, dim=1)
        mask[:, distances <= dropout_radius] = -float("inf")
    return A + mask
