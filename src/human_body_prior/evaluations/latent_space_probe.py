# -*- coding: utf-8 -*-
"""Encode VPoser dataset poses and visualize the latent space.

Example:
    python -m human_body_prior.evaluations.latent_space_probe \
        --expr-dir support_data/training/training_experiments/V02_05 \
        --dataset-dir AMASS/DataSet/V02_05 \
        --split vald \
        --6DOF False \
        --out-dir latent_probe_out
"""

import argparse
import ast
import configparser
import glob
import importlib.util
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.models.vposer_model import VPoser, VPoserOld
from human_body_prior.tools.model_loader import load_model


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
DEFAULT_VPOSER_V1_DIR = osp.join(REPO_ROOT, "support_data", "dowloads", "vposer_v1_0")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def pose_magnitude_feature(pose_body):
    """Simple built-in feature: mean axis-angle magnitude per pose."""
    pose_body = pose_body.view(pose_body.shape[0], 21, 3)
    return torch.linalg.norm(pose_body, dim=-1).mean(dim=-1)


def pose_delta_feature(pose_body, previous_pose_body=None):
    """Frame-to-frame pose change proxy; only meaningful if data order is temporal."""
    pose_body = pose_body.view(pose_body.shape[0], -1)
    if previous_pose_body is None:
        delta = torch.zeros(pose_body.shape[0], device=pose_body.device, dtype=pose_body.dtype)
        delta[1:] = torch.linalg.norm(pose_body[1:] - pose_body[:-1], dim=-1)
        return delta

    previous_pose_body = previous_pose_body.view(1, -1).to(pose_body)
    combined = torch.cat([previous_pose_body, pose_body], dim=0)
    return torch.linalg.norm(combined[1:] - combined[:-1], dim=-1)


def feature_from_batch(batch, pose_body, feature_name, previous_pose_body):
    if feature_name == "pose_magnitude":
        return pose_magnitude_feature(pose_body)
    if feature_name == "pose_delta":
        return pose_delta_feature(pose_body, previous_pose_body)
    if feature_name in batch:
        return batch[feature_name].to(pose_body.device).view(-1)
    raise ValueError("Feature '{}' was not found in the dataset batch.".format(feature_name))


def _load_vposer_v1(expr_dir, device):
    """Load the packaged vposer_v1_0 model from support_data/dowloads."""
    model_code_path = osp.join(expr_dir, "vposer_smpl.py")
    cfg_paths = glob.glob(osp.join(expr_dir, "*.ini"))
    snapshot_paths = sorted(glob.glob(osp.join(expr_dir, "snapshots", "*.pt")), key=osp.getmtime)

    if not osp.exists(model_code_path):
        raise ValueError("Could not find legacy model code: {}".format(model_code_path))
    if not cfg_paths:
        raise ValueError("Could not find legacy VPoser .ini config in: {}".format(expr_dir))
    if not snapshot_paths:
        raise ValueError("Could not find legacy VPoser .pt snapshot in: {}".format(osp.join(expr_dir, "snapshots")))

    spec = importlib.util.spec_from_file_location("legacy_vposer_smpl", model_code_path)
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)

    cfg = configparser.ConfigParser()
    cfg.read(cfg_paths[0])
    params = cfg["All"]

    model = legacy_module.VPoser(
        num_neurons=params.getint("num_neurons"),
        latentD=params.getint("latentD"),
        data_shape=ast.literal_eval(params.get("data_shape")),
        use_cont_repr=params.getboolean("use_cont_repr"),
    )

    load_kwargs = {"map_location": torch.device("cpu")} if device == "cpu" else {}
    weights = torch.load(snapshot_paths[-1], **load_kwargs)
    state_dict = weights["state_dict"] if isinstance(weights, dict) and "state_dict" in weights else weights
    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_branch_model(expr_dir, device, model_variant):
    if expr_dir == DEFAULT_VPOSER_V1_DIR:
        raise ValueError(
            "--6DOF True needs --expr-dir to point to a trained 6DOF experiment "
            "directory with snapshots/*.ckpt, not the packaged vposer_v1_0 model."
        )

    model_code = VPoserOld if model_variant == "old" else VPoser
    vp_model, _ = load_model(
        expr_dir=expr_dir,
        model_code=model_code,
        remove_words_in_model_weights="vp_model.",
        comp_device=device,
    )
    return vp_model


def encode_dataset(expr_dir, dataset_dir, split, batch_size, max_batches, device, model_variant, feature_name):
    if model_variant == "v1":
        vp_model = _load_vposer_v1(expr_dir, device)
    else:
        vp_model = _load_branch_model(expr_dir, device, model_variant)

    vp_model = vp_model.to(device).eval()

    data_fields = ["pose_body"] if feature_name in ("pose_magnitude", "pose_delta") else ["pose_body", feature_name]
    ds = VPoserDS(osp.join(dataset_dir, split), data_fields=data_fields)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    latents, features = [], []
    previous_pose_body = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            pose_body = batch["pose_body"].to(device).view(-1, 63)
            encoded = vp_model.encode(pose_body)

            # vposer_v1_0 and original VPoser return Normal(...) directly.
            # Current 6D branch returns {"latents": Normal(...)}.
            q_z = encoded["latents"] if isinstance(encoded, dict) else encoded
            latents.append(q_z.mean.detach().cpu())
            features.append(feature_from_batch(batch, pose_body, feature_name, previous_pose_body).detach().cpu())
            previous_pose_body = pose_body[-1:].detach()

    return torch.cat(latents).numpy(), torch.cat(features).numpy()


def project_latents(latents, method):
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(latents)

    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=13).fit_transform(latents)

    raise ValueError("Unknown projection method: {}".format(method))


def save_plot(points_2d, feature, out_png, title):
    plt.figure(figsize=(8, 6), dpi=150)
    sc = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=feature, s=6, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="feature value")
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    
    print(f"Plot saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expr-dir",
        default=DEFAULT_VPOSER_V1_DIR,
        help="VPoser experiment directory. Defaults to support_data/dowloads/vposer_v1_0.",
    )
    parser.add_argument("--dataset-dir", required=True, help="Prepared VPoser dataset root containing train/vald/test")
    parser.add_argument("--split", default="vald", choices=["train", "vald", "test"])
    parser.add_argument("--out-dir", default="latent_probe_out")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--method", default="pca", choices=["pca", "umap"])
    parser.add_argument(
        "--feature",
        default="pose_magnitude",
        help="Feature used to color points. Use pose_magnitude, pose_delta, or a dataset field such as pose_speed.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--6DOF",
        dest="six_dof",
        type=str2bool,
        default=False,
        help="Set True to load the 6DOF-input VPoser. Set False for original/pretrained VPoser.",
    )
    parser.add_argument(
        "--model-variant",
        default="v1",
        choices=["v1", "old", "6d"],
        help="Use v1 for support_data/dowloads/vposer_v1_0, old for a branch .ckpt original model, or 6d for this branch's 6D-input model.",
    )
    args = parser.parse_args()

    if args.six_dof:
        args.model_variant = "6d"

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    latents, feature = encode_dataset(
        args.expr_dir,
        args.dataset_dir,
        args.split,
        args.batch_size,
        args.max_batches,
        device,
        args.model_variant,
        args.feature,
    )
    points_2d = project_latents(latents, args.method)

    np.savez(
        osp.join(args.out_dir, "latent_probe_{}.npz".format(args.split)),
        latents=latents,
        feature=feature,
        points_2d=points_2d,
    )
    save_plot(
        points_2d,
        feature,
        osp.join(args.out_dir, "latent_probe_{}_{}.png".format(args.split, args.method)),
        "VPoser latent space: {} colored by {}".format(args.split, args.feature),
    )


if __name__ == "__main__":
    main()
