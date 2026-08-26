"""
Training entry point reproducing all three of the original notebooks
(baseline270.ipynb, br6_sp3d_base.ipynb, br6_sp3d_270.ipynb) as named
`--variant` configs, sharing the model/training code in `common.py` and
`spatial_attention.py`.

Also drops one dead cell from br6_sp3d_270.ipynb that looked like leftover
scratch debugging and would raise on execution:

    C=207
    T=750
    z = torch.random (C,T,1)          # torch.random is a module, not callable like this
    x = SpatialAttentionLayer.forward()  # unbound method call with no args/self

Usage:
    export COMET_API_KEY=...          # optional, omit --log-to-comet to skip
    python train.py --variant harmonics_270 --data-dir /path/to/MASC-MEG/process_v2/
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import BrainModule, DoubleDataset, Trainer, count_parameters
from spatial_attention import FourierSpatialAttention, HarmonicsSpatialAttention

CONFIGS = {
    # Reproduces baseline270.ipynb: 2D Fourier spatial attention, no parameter reduction.
    "baseline": dict(
        attention="fourier", n_channels_attention=270, n_channels_unmix=270, n_spatial_harmonics=32,
        checkpoint="baseline270", default_nepoch=10,
    ),
    # Reproduces br6_sp3d_base.ipynb: spherical-harmonics attention, smaller unmix width.
    "harmonics_base": dict(
        attention="harmonics", n_channels_attention=270, n_channels_unmix=5, n_spatial_harmonics=24,
        checkpoint="br6_sp3d_base_temperature", default_nepoch=10,
    ),
    # Reproduces br6_sp3d_270.ipynb: spherical-harmonics attention, the configuration reported
    # in the README/publication (~40% parameter reduction vs. baseline).
    "harmonics_270": dict(
        attention="harmonics", n_channels_attention=270, n_channels_unmix=6, n_spatial_harmonics=32,
        checkpoint="br6_sp3d_base_temperature", default_nepoch=20,
    ),
}

COMMON_HPARAMS = dict(
    batch_size=100, lr_fe=3e-4, use_scheduler=False, scheduler_rate=0.95, optim="AdamW", weight_decay=1e-1,
    clip_temperature=1, clip_temperature_type="param", clip_temperature_lr=1e-3, meg_sr=100, meg_offset=0.15,
    hidden="extract_features", n_subjects=27, n_channels_input=208, spatial_dropout_number=0,
    spatial_dropout_radius=0.1, use_unmixing_layer=True, use_subject_layer=True, regularize_subject_layer=False,
    bias_subject_layer=False, n_channels_block=320, n_blocks=5, head_pool="conv", n_features=512,
    loss="clip", head_stride=2,
)


def build_model(variant_config, hparam, data_dir, coords_xy_scaled):
    if variant_config["attention"] == "fourier":
        attention_layer = FourierSpatialAttention(
            n_input=hparam["n_channels_input"], n_output=hparam["n_channels_attention"],
            K=hparam["n_spatial_harmonics"], coords_xy=coords_xy_scaled,
            n_dropout=hparam["spatial_dropout_number"], dropout_radius=hparam["spatial_dropout_radius"],
        )
    else:
        attention_layer = HarmonicsSpatialAttention(
            n_input=hparam["n_channels_input"], n_output=hparam["n_channels_attention"],
            K=hparam["n_spatial_harmonics"], coords_xy=coords_xy_scaled,
            n_dropout=hparam["spatial_dropout_number"], dropout_radius=hparam["spatial_dropout_radius"],
            dirprocess=data_dir,
        )

    return BrainModule(
        spatial_attention_layer=attention_layer,
        n_channels_input=hparam["n_channels_input"],
        n_channels_attention=hparam["n_channels_attention"],
        n_channels_unmix=hparam["n_channels_unmix"],
        use_unmixing_layer=hparam["use_unmixing_layer"],
        use_subject_layer=hparam["use_subject_layer"],
        n_subjects=hparam["n_subjects"],
        regularize_subject_layer=hparam["regularize_subject_layer"],
        bias_subject_layer=hparam["bias_subject_layer"],
        n_channels_block=hparam["n_channels_block"],
        n_features=hparam["n_features"],
        head_pool=hparam["head_pool"],
        head_stride=hparam["head_stride"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=CONFIGS.keys(), required=True)
    parser.add_argument("--data-dir", required=True, help="Path to the processed MASC-MEG dataset (dirprocess)")
    parser.add_argument("--nepoch", type=int, default=None, help="Defaults to the value used for this variant's reported result")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-to-comet", action="store_true", help="Log metrics to Comet ML (needs COMET_API_KEY)")
    parser.add_argument("--comet-workspace", default=None)
    args = parser.parse_args()

    variant = CONFIGS[args.variant]
    hparam = {**COMMON_HPARAMS, **{k: v for k, v in variant.items() if k not in ("default_nepoch", "attention")},
              "dirprocess": args.data_dir}
    if hparam["hidden"] == "lms":
        hparam["n_features"], hparam["head_stride"] = 120, 1
    nepoch = args.nepoch or variant["default_nepoch"]

    coords_xy_scaled = np.load(os.path.join(args.data_dir, "coords/coords208_xy_scaled.npy"))
    hidden_train = np.load(os.path.join(args.data_dir, f"audio/{hparam['hidden']}_train4.npy"))
    hidden_test = np.load(os.path.join(args.data_dir, f"audio/{hparam['hidden']}_test4.npy"))
    df_train = pd.read_csv(os.path.join(args.data_dir, f"dataframe/df_train{hparam['n_subjects']}.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, f"dataframe/df_test{hparam['n_subjects']}.csv"))
    meg = dict(np.load(os.path.join(
        args.data_dir, f"meg/meg{hparam['n_subjects']}_sr{hparam['meg_sr']}_default_v1.npz"
    )))

    dataset_train = DoubleDataset(meg, hidden_train, df_train, hparam["meg_sr"], hparam["meg_offset"])
    dataset_test = DoubleDataset(meg, hidden_test, df_test, hparam["meg_sr"], hparam["meg_offset"])
    dataloader_train = DataLoader(dataset_train, batch_size=hparam["batch_size"], shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=hparam["batch_size"] // 5, shuffle=False)

    model = build_model(variant, hparam, args.data_dir, coords_xy_scaled)
    print(f"Trainable parameters ({args.variant}): {count_parameters(model):,}")
    print(f"  of which in the spatial module: {count_parameters(model.spatial_module):,}")

    experiment = None
    if args.log_to_comet:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name=f"metameg{hparam['n_subjects']}",
            workspace=args.comet_workspace,
            auto_output_logging=False,
        )
        experiment.log_parameters(hparam)
        experiment.log_code()

    trainer = Trainer(model, hparam, experiment=experiment)
    try:
        trainer.fit(dataloader_train, dataloader_test, hidden_test, nepoch=nepoch, device=args.device)
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
