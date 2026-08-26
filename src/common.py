"""
Shared model components for the MEG-to-speech representation model
(spatial attention -> per-subject unmixing -> temporal conv stack ->
feature projection, trained with a CLIP-style contrastive loss against
audio/speech embeddings), following Défossez et al., *Decoding speech
perception from non-invasive brain recordings*, Nat. Mach. Intell. 2023.

This file merges what were three ~95%-identical notebooks
(baseline270.ipynb, br6_sp3d_base.ipynb, br6_sp3d_270.ipynb) into one
shared module. The only real difference between the three was which
spatial-attention layer was used (see `spatial_attention.py`) and a
handful of hyperparameters — see `train.py`, which reproduces all three
as named configs.

Fixes applied while merging (see inline comments for exact locations):
  1. `Trainer.fit`'s validation loop referenced the bare `model` global
     instead of `self.model` — harmless only by accident (the notebook
     always passed the same object to both names), but would silently
     validate the wrong model if `Trainer` were ever reused.
  2. The "save best checkpoint" logic had `val_loss_min` and `val_loss`
     swapped: `if val_loss < val_loss_min: ...; val_loss = val_loss_min`
     never updates `val_loss_min`, so the `if` was always true and every
     single epoch got saved as if it were the best one. Fixed to
     `val_loss_min = val_loss`.
  3. `metrics()` normalized with `dim=(--1)` (double-negation of 1, i.e.
     `dim=1`) instead of `dim=(-1)` in the 2D branch — for a 2D tensor
     these happen to be the same axis, so it wasn't a live bug, but it
     was clearly a typo. Fixed to `dim=(-1)`.
  4. Renamed the `n_attantion` / `n_channels_attantion` typo to
     `n_attention` / `n_channels_attention` throughout.
"""

import math
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


def cart2sph(sensor_xyz):
    """Cartesian (x, y, z) sensor coordinates -> spherical (r, theta, phi)."""
    x, y, z = sensor_xyz[:, 0], sensor_xyz[:, 1], sensor_xyz[:, 2]
    xy = np.linalg.norm(sensor_xyz[:, :2], axis=-1)
    r = np.linalg.norm(sensor_xyz, axis=-1)
    theta = np.arctan2(xy, z)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)


class CLIPloss(nn.Module):
    """Symmetric contrastive loss (CLIP-style) between brainwave and audio embeddings, with a learnable temperature."""

    def __init__(self, clip_temperature, clip_temperature_type):
        super().__init__()
        if clip_temperature_type == "param":
            self.temperature = nn.Parameter(torch.tensor(math.log(clip_temperature), dtype=torch.float32))
        elif clip_temperature_type == "hparam":
            self.temperature = torch.tensor(math.log(clip_temperature), dtype=torch.float32, requires_grad=False)
        else:
            raise ValueError("clip_temperature_type must be 'param' or 'hparam'")

    def forward(self, brainwave_embeddings, audio_embeddings):
        batch_size = brainwave_embeddings.size(0)
        if audio_embeddings.dim() == 3:
            brainwave_embeddings = F.normalize(brainwave_embeddings, dim=(-2, -1))
            audio_embeddings = F.normalize(audio_embeddings, dim=(-2, -1))
            similarity = torch.einsum("Bef, bef -> Bb", brainwave_embeddings, audio_embeddings)
        else:
            brainwave_embeddings = F.normalize(brainwave_embeddings, dim=-1)
            audio_embeddings = F.normalize(audio_embeddings, dim=-1)
            similarity = torch.einsum("Bf, bf -> Bb", brainwave_embeddings, audio_embeddings)

        labels = torch.arange(batch_size, device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        loss_temperature = F.cross_entropy(similarity / torch.exp(self.temperature), labels)
        return loss, loss_temperature


class MSE(nn.Module):
    """Plain MSE loss between flattened brainwave and audio embeddings (alternative to CLIPloss)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, brainwave_embeddings, audio_embeddings, temperature=None):
        brainwave_embeddings = einops.rearrange(brainwave_embeddings, "b f t -> b (f t)")
        audio_embeddings = einops.rearrange(audio_embeddings, "b f t -> b (f t)")
        return self.mse(brainwave_embeddings, audio_embeddings)


def metrics(brainwave_embeddings, audio_embeddings, labels):
    """Top-1 / top-10 retrieval accuracy of the correct audio embedding among the batch."""
    if audio_embeddings.dim() == 3:
        brainwave_embeddings = F.normalize(brainwave_embeddings, dim=(-2, -1))
        audio_embeddings = F.normalize(audio_embeddings, dim=(-2, -1))
        similarity = torch.einsum("Bef, bef -> Bb", brainwave_embeddings, audio_embeddings)
    else:
        brainwave_embeddings = F.normalize(brainwave_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        similarity = torch.einsum("Bf, bf -> Bb", brainwave_embeddings, audio_embeddings)

    labels = labels.view(-1, 1)
    top10 = torch.topk(similarity, 10, dim=-1).indices
    top1 = torch.topk(similarity, 1, dim=-1).indices
    return torch.eq(top10, labels).any(dim=1), torch.eq(top1, labels).any(dim=1)


class SubjectPlusLayer(nn.Module):
    """Per-subject 1x1-conv 'unmixing' layer: a learned linear map from attention-space channels to a shared space, one map per subject."""

    def __init__(self, n_input, n_output, n_subjects, regularize=True, bias=False):
        super().__init__()
        self.bias = bias
        self.regularize = regularize
        self.regularizer = None

        A, b = self._create_parameters(n_input, n_output, n_subjects)
        self.A = nn.Parameter(A)
        self.register_buffer("I", torch.zeros((1, n_output, n_input), requires_grad=False))
        if self.bias:
            self.b = nn.Parameter(b)
            self.register_buffer("zero", torch.zeros(size=(1, n_output, 1)))

    def _create_parameters(self, n_input, n_output, n_subjects):
        A = torch.zeros(size=(n_subjects, n_output, n_input))
        b = torch.zeros(size=(n_subjects, n_output, 1)) if self.bias else None
        with torch.no_grad():
            for subject in range(n_subjects):
                layer = nn.Conv1d(in_channels=n_input, out_channels=n_output, kernel_size=1)
                A[subject] = einops.rearrange(layer.weight.data, "o i 1 -> o i")
                if self.bias:
                    b[subject] = einops.rearrange(layer.bias.data, "o -> o 1")
        return A, b

    def _create_regularizer(self, A, b):
        batch_size = A.shape[0]
        reg = torch.norm(A - self.I, p="fro")
        if self.bias:
            reg = reg + torch.norm(b, p="fro")
        return reg / batch_size

    def get_regularizer(self):
        regularizer, self.regularizer = self.regularizer, None
        return regularizer

    def forward(self, x, subject_ids):
        A = torch.cat([self.I, self.A], dim=0)
        subject_ids = subject_ids.clone()
        subject_ids[subject_ids >= A.size(0)] = 0
        A_selected = A[subject_ids, :, :]
        out = torch.einsum("bji, bit -> bjt", A_selected, x)

        b_selected = None
        if self.bias:
            b = torch.cat([self.zero, self.b], dim=0)
            b_selected = b[subject_ids, :, :]
            out = out + b_selected

        if self.regularize and self.training:
            self.regularizer = self._create_regularizer(A_selected, b_selected)
        return out


class ConvBlock(nn.Module):
    """Dilated residual 1D conv block (3 convs, GELU/GLU activations) — the repeating unit of the temporal stack."""

    def __init__(self, n_input, n_output, block_index):
        super().__init__()
        kernel_size = 3
        self.block_index = block_index
        dilation1 = 2 ** (2 * block_index % 5)
        dilation2 = 2 ** ((2 * block_index + 1) % 5)
        dilation3 = 2

        self.conv1 = nn.Conv1d(n_input, n_output, kernel_size, dilation=dilation1, padding="same")
        self.conv2 = nn.Conv1d(n_output, n_output, kernel_size, dilation=dilation2, padding="same")
        self.conv3 = nn.Conv1d(n_output, 2 * n_output, kernel_size, dilation=dilation3, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(n_output)
        self.batchnorm2 = nn.BatchNorm1d(n_output)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.activation3 = nn.GLU(dim=-2)

    def forward(self, x):
        res1 = self.conv1(x) if self.block_index == 0 else x + self.conv1(x)
        res1 = self.activation1(self.batchnorm1(res1))

        res2 = res1 + self.conv2(res1)
        res2 = self.activation2(self.batchnorm2(res2))

        return self.activation3(self.conv3(res2))


class ConvHead(nn.Module):
    """Downsamples the temporal stack's output to the final feature dimension."""

    def __init__(self, n_channels, n_features, pool, head_stride):
        super().__init__()
        if pool == "max":
            self.pool = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=head_stride, padding=0 if head_stride == 2 else 1),
                nn.Conv1d(n_channels, 2 * n_channels, kernel_size=1),
            )
        elif pool == "conv":
            self.pool = nn.Conv1d(n_channels, 2 * n_channels, kernel_size=3, stride=head_stride,
                                   padding=0 if head_stride == 2 else 1)
        else:
            raise ValueError("pool must be 'max' or 'conv'")

        self.conv = nn.Conv1d(2 * n_channels, n_features, kernel_size=1)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = self.activation(self.pool(x))
        return self.batch_norm(self.conv(x))


class SpatialModule(nn.Module):
    """Spatial attention (a `spatial_attention.SpatialAttention*` instance) -> optional unmixing conv -> per-subject layer."""

    def __init__(self, n_input, n_attention, n_unmix, spatial_attention_layer, use_unmixing_layer,
                 use_subject_layer, n_subjects, regularize_subject_layer, bias_subject_layer):
        super().__init__()
        self.self_attention = spatial_attention_layer  # None to skip spatial attention entirely

        if use_unmixing_layer:
            n_attention = n_attention if self.self_attention else n_input
            self.unmixing_layer = nn.Conv1d(n_attention, n_attention, kernel_size=1)
        else:
            self.unmixing_layer = None

        if use_subject_layer:
            n_attention = n_attention if (self.self_attention or self.unmixing_layer) else n_input
            self.subject_layer = SubjectPlusLayer(n_attention, n_unmix, n_subjects,
                                                   regularize=regularize_subject_layer, bias=bias_subject_layer)
        else:
            self.subject_layer = None

    def forward(self, xs):
        x, subject_ids = xs
        if self.self_attention:
            x = self.self_attention(x)
        if self.unmixing_layer:
            x = self.unmixing_layer(x)
        if self.subject_layer:
            x = self.subject_layer(x, subject_ids)
        return x


class TemporalModule(nn.Module):
    """Stack of 5 `ConvBlock`s."""

    def __init__(self, n_unmix, n_block, n_blocks=5):
        super().__init__()
        self.conv_blocks = nn.ModuleDict({
            f"conv_block_{i}": ConvBlock(n_unmix if i == 0 else n_block, n_block, i)
            for i in range(n_blocks)
        })

    def forward(self, x):
        for block in self.conv_blocks.values():
            x = block(x)
        return x


class BrainModule(nn.Module):
    """Full model: SpatialModule -> TemporalModule -> ConvHead."""

    def __init__(self, spatial_attention_layer, n_channels_input, n_channels_attention, n_channels_unmix,
                 use_unmixing_layer, use_subject_layer, n_subjects, regularize_subject_layer, bias_subject_layer,
                 n_channels_block, n_features, head_pool, head_stride, **_ignored):
        super().__init__()
        self.spatial_module = SpatialModule(
            n_input=n_channels_input, n_attention=n_channels_attention, n_unmix=n_channels_unmix,
            spatial_attention_layer=spatial_attention_layer, use_unmixing_layer=use_unmixing_layer,
            use_subject_layer=use_subject_layer, n_subjects=n_subjects,
            regularize_subject_layer=regularize_subject_layer, bias_subject_layer=bias_subject_layer,
        )
        self.temporal_module = TemporalModule(n_unmix=n_channels_unmix, n_block=n_channels_block)
        self.feature_projection = ConvHead(n_channels_block, n_features, head_pool, head_stride)

    def forward(self, xs):
        z = self.spatial_module(xs)
        y = self.feature_projection(self.temporal_module(z))
        return z, y


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class DoubleDataset(TorchDataset):
    """Pairs an MEG segment with its corresponding audio/speech hidden-state embedding."""

    def __init__(self, meg, hidden, df, meg_sr, meg_offset=0):
        self.meg = meg
        self.hidden = hidden
        self.df = df
        self.meg_sr = meg_sr
        self.meg_offset = int(meg_offset * meg_sr)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.loc[index]
        subject_id = str(row["subject_id"])
        subject_id = subject_id.zfill(2)
        subset = f"subject{subject_id}_session{row['session_id']}_story{row['story_id']}"

        start = row[f"meg{self.meg_sr}_start"] + self.meg_offset
        stop = row[f"meg{self.meg_sr}_stop"] + self.meg_offset
        wav_index = row["wav_index"]

        meg = torch.tensor(self.meg[subset][:, start:stop], dtype=torch.float32)
        hidden = torch.tensor(self.hidden[wav_index], dtype=torch.float32)
        return meg, torch.tensor(row["subject_id"], dtype=torch.long), hidden, torch.tensor(wav_index, dtype=torch.long)


class Trainer:
    """Training loop with CLIP/MSE loss, optional Comet ML logging, and best-checkpoint saving."""

    def __init__(self, model, hparam, experiment=None):
        self.model = model
        self.experiment = experiment

        if hparam["loss"] == "clip":
            self.criterion = CLIPloss(hparam["clip_temperature"], hparam["clip_temperature_type"])
        elif hparam["loss"] == "mse":
            self.criterion = MSE()
        else:
            raise ValueError("hparam['loss'] must be 'clip' or 'mse'")

        parameters = [
            {"params": self.model.spatial_module.parameters(), "lr": hparam["lr_fe"], "weight_decay": hparam["weight_decay"]},
            {"params": self.model.temporal_module.parameters(), "lr": hparam["lr_fe"], "weight_decay": hparam["weight_decay"]},
            {"params": self.model.feature_projection.parameters(), "lr": hparam["lr_fe"], "weight_decay": hparam["weight_decay"]},
            {"params": self.criterion.parameters(), "lr": hparam["clip_temperature_lr"], "weight_decay": 0},
        ]
        optim_cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[hparam["optim"]]
        self.optimizer = optim_cls(parameters)
        self.scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=hparam["scheduler_rate"])
            if hparam["use_scheduler"] else None
        )
        self.checkpoint_name = hparam["checkpoint"]

    def fit(self, dataloader_train, dataloader_val, hidden_test, nepoch, device):
        hidden_test = einops.rearrange(
            torch.tensor(hidden_test, dtype=torch.float32).to(device), "b t f -> b f t"
        )
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        val_loss_min = float("inf")

        for epoch in range(nepoch):
            train_loss = self._train_one_epoch(dataloader_train, device, epoch)
            if self.experiment is not None:
                self.experiment.log_metric("loss_train", train_loss, step=epoch)
            if self.scheduler:
                self.scheduler.step()

            val_loss, top10, top1 = self._validate_one_epoch(dataloader_val, hidden_test, device, epoch)
            if self.experiment is not None:
                self.experiment.log_metric("loss_test", val_loss, step=epoch)
                self.experiment.log_metric("top10s_test", top10, step=epoch)
                self.experiment.log_metric("top1s_test", top1, step=epoch)

            if val_loss < val_loss_min:
                val_loss_min = val_loss  # was `val_loss = val_loss_min` in the original — see module docstring
                os.makedirs("checkpoint", exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": val_loss,
                }, f"checkpoint/{self.checkpoint_name}.pt")

    def _train_one_epoch(self, dataloader, device, epoch):
        self.model.train()
        self.criterion.train()
        train_loss = 0.0
        n_samples = 0
        for meg, sbj, hidden, _ in dataloader:
            hidden = einops.rearrange(hidden, "b t f -> b f t")
            meg, sbj, hidden = meg.to(device), sbj.to(device), hidden.to(device)

            _, result = self.model((meg, sbj))
            _, loss_temperature = self.criterion(result, hidden)

            self.optimizer.zero_grad()
            loss_temperature.backward()
            self.optimizer.step()
            train_loss += loss_temperature.item() * meg.shape[0]
            n_samples += meg.shape[0]
        return train_loss / max(n_samples, 1)

    def _validate_one_epoch(self, dataloader, hidden_test, device, epoch):
        self.model.eval()
        self.criterion.eval()
        val_loss, n_samples = 0.0, 0
        top10_hits, top1_hits = [], []
        with torch.no_grad():
            for meg, sbj, hidden, widx in dataloader:
                hidden = einops.rearrange(hidden, "b t f -> b f t")
                meg, sbj, hidden, widx = meg.to(device), sbj.to(device), hidden.to(device), widx.to(device)

                _, result = self.model((meg, sbj))
                _, loss_temperature = self.criterion(result, hidden)
                val_loss += loss_temperature.item() * meg.shape[0]
                n_samples += meg.shape[0]

                is_top10, is_top1 = metrics(result, hidden_test, widx)
                top10_hits.append(is_top10.cpu().numpy())
                top1_hits.append(is_top1.cpu().numpy())

        return (
            val_loss / max(n_samples, 1),
            float(np.mean(np.concatenate(top10_hits))),
            float(np.mean(np.concatenate(top1_hits))),
        )
