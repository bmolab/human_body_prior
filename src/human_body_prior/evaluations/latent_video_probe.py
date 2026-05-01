# -*- coding: utf-8 -*-
"""Encode one ordered motion sequence and visualize its VPoser latent trajectory.

Example:
    python -m human_body_prior.evaluations.latent_video_probe \
        --npz AMASS/SFU/0005/0005_Walking001_poses.npz \
        --6DOF False \
        --method pca \
        --out-dir video_probe_walking
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


def _load_vposer_v1(expr_dir, device):
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


def _load_branch_model(expr_dir, device, six_dof):
    if expr_dir == DEFAULT_VPOSER_V1_DIR:
        raise ValueError(
            "--6DOF True needs --expr-dir to point to a trained 6DOF experiment "
            "directory with snapshots/*.ckpt, not the packaged vposer_v1_0 model."
        )

    model_code = VPoser if six_dof else VPoserOld
    vp_model, _ = load_model(
        expr_dir=expr_dir,
        model_code=model_code,
        remove_words_in_model_weights="vp_model.",
        comp_device=device,
    )
    return vp_model


def load_vposer_model(expr_dir, device, six_dof):
    if six_dof:
        model = _load_branch_model(expr_dir, device, six_dof=True)
    elif expr_dir == DEFAULT_VPOSER_V1_DIR:
        model = _load_vposer_v1(expr_dir, device)
    else:
        model = _load_branch_model(expr_dir, device, six_dof=False)

    return model.to(device).eval()


def load_pose_body_from_npz(npz_path, start_frame=None, end_frame=None, stride=1):
    data = np.load(npz_path)
    if "poses" not in data:
        raise ValueError("Expected AMASS-style key 'poses' in {}".format(npz_path))

    poses = data["poses"].astype(np.float32)
    pose_body = poses[:, 3:66]
    pose_body = pose_body[start_frame:end_frame:stride]

    if len(pose_body) == 0:
        raise ValueError("No frames selected from {}".format(npz_path))
    return torch.tensor(pose_body, dtype=torch.float32)


def encode_pose_sequence(vp_model, pose_body, batch_size, device):
    latents = []
    with torch.no_grad():
        for start in range(0, len(pose_body), batch_size):
            batch = pose_body[start:start + batch_size].to(device)
            encoded = vp_model.encode(batch)
            q_z = encoded["latents"] if isinstance(encoded, dict) else encoded
            latents.append(q_z.mean.detach().cpu())
    return torch.cat(latents).numpy()


def pose_speed_feature(pose_body):
    pose_body = pose_body.view(pose_body.shape[0], -1)
    speed = torch.zeros(pose_body.shape[0], dtype=pose_body.dtype)
    speed[1:] = torch.linalg.norm(pose_body[1:] - pose_body[:-1], dim=-1)
    return speed.numpy()


def project_latents(latents, method):
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(latents)

    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=13).fit_transform(latents)

    raise ValueError("Unknown projection method: {}".format(method))


def save_feature_over_time(feature, out_png):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(np.arange(len(feature)), feature, linewidth=1.5)
    ax.set_title("Pose speed over time")
    ax.set_xlabel("frame")
    ax.set_ylabel("pose speed")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_latent_trajectory(points_2d, color_values, out_png, title, colorbar_label):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sc = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=color_values, s=10, cmap="viridis", alpha=0.9)
    ax.plot(points_2d[:, 0], points_2d[:, 1], color="0.65", linewidth=0.8, alpha=0.7)
    ax.scatter(points_2d[0, 0], points_2d[0, 1], marker="o", s=60, color="green", label="start")
    ax.scatter(points_2d[-1, 0], points_2d[-1, 1], marker="x", s=60, color="red", label="end")
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="AMASS sequence .npz containing a poses array.")
    parser.add_argument(
        "--expr-dir",
        default=DEFAULT_VPOSER_V1_DIR,
        help="VPoser experiment directory. Defaults to support_data/dowloads/vposer_v1_0.",
    )
    parser.add_argument("--out-dir", default="latent_video_probe_out")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--method", default="pca", choices=["pca", "umap"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--6DOF", dest="six_dof", type=str2bool, default=False)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    vp_model = load_vposer_model(args.expr_dir, device, args.six_dof)
    pose_body = load_pose_body_from_npz(args.npz, args.start_frame, args.end_frame, args.stride)
    latents = encode_pose_sequence(vp_model, pose_body, args.batch_size, device)
    pose_speed = pose_speed_feature(pose_body)
    points_2d = project_latents(latents, args.method)
    frame_ids = np.arange(len(pose_body))

    np.savez(
        osp.join(args.out_dir, "latent_video_probe.npz"),
        latents=latents,
        pose_speed=pose_speed,
        points_2d=points_2d,
        frame_ids=frame_ids,
        source_npz=args.npz,
    )

    save_feature_over_time(pose_speed, osp.join(args.out_dir, "feature_over_time.png"))
    save_latent_trajectory(
        points_2d,
        frame_ids,
        osp.join(args.out_dir, "latent_trajectory_time.png"),
        "VPoser latent trajectory colored by time",
        "frame",
    )
    save_latent_trajectory(
        points_2d,
        pose_speed,
        osp.join(args.out_dir, "latent_trajectory_feature.png"),
        "VPoser latent trajectory colored by pose speed",
        "pose speed",
    )


if __name__ == "__main__":
    main()
