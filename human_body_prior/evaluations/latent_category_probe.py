"""Visualize VPoser latent categories inferred from motion filenames.

Each plotted point is one sampled pose frame from one AMASS-style .npz motion
file. The point position is the 2D projection of that frame's VPoser latent
code, and the color is the motion category parsed from the filename.

Example:
    python -m human_body_prior.evaluations.latent_category_probe \
        --data-dir AMASS/BioMotionLab_NTroje \
        --6DOF False \
        --method umap \
        --frames-per-file 100 \
        --out-dir latent_probe/biomotion_categories
"""

import argparse
import csv
import os
import os.path as osp
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from human_body_prior.evaluations.latent_video_probe import (
    DEFAULT_VPOSER_V1_DIR,
    load_vposer_model,
    project_latents,
    str2bool,
)


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def progress_iter(iterable, desc=None, total=None, enabled=True):
    """Use tqdm when available, otherwise return the original iterable."""
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total)
    return iterable


def find_npz_files(data_dir, max_files=None):
    """Return AMASS-style pose files below data_dir in stable sorted order."""
    npz_files = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith("_poses.npz"):
                npz_files.append(osp.join(root, fname))

    npz_files = sorted(npz_files)
    if max_files is not None:
        npz_files = npz_files[:max_files]
    return npz_files


def parse_category(npz_path, category_regex=None):
    """Infer a motion category from a filename.

    Default examples:
        0005_normal_walk1_poses.npz -> normal_walk
        0014_lifting_light2_poses.npz -> lifting_light
        0032_some_new_category7_poses.npz -> some_new_category
    """
    stem = osp.splitext(osp.basename(npz_path))[0]

    if category_regex is not None:
        match = re.search(category_regex, stem)
        if match is None:
            raise ValueError(
                "Category regex did not match filename '{}': {}".format(
                    stem, category_regex
                )
            )
        return match.group(1) if match.groups() else match.group(0)

    category = re.sub(r"_poses$", "", stem)
    category = re.sub(r"^\d+_", "", category)
    category = re.sub(r"\d+$", "", category)
    category = category.strip("_").lower()
    return category if category else "unknown"


def selected_frame_ids(num_frames, skip_edge_ratio, stride, frames_per_file, sample_mode):
    """Choose frame ids from one motion file."""
    start = int(round(skip_edge_ratio * num_frames))
    end = int(round((1.0 - skip_edge_ratio) * num_frames))
    start = max(0, min(start, num_frames))
    end = max(start, min(end, num_frames))

    ids = np.arange(start, end, max(1, stride), dtype=np.int64)
    if len(ids) == 0:
        return ids

    if frames_per_file is None or len(ids) <= frames_per_file:
        return ids

    if sample_mode == "even":
        chosen = np.linspace(0, len(ids) - 1, frames_per_file).round().astype(np.int64)
        return ids[np.unique(chosen)]

    if sample_mode == "random":
        chosen = np.random.choice(len(ids), size=frames_per_file, replace=False)
        return np.sort(ids[chosen])

    raise ValueError("Unknown sample mode: {}".format(sample_mode))


def load_sampled_poses(
    npz_files,
    frames_per_file,
    skip_edge_ratio,
    stride,
    sample_mode,
    category_regex,
):
    """Load sampled pose_body frames and metadata from many motion files."""
    pose_chunks = []
    categories = []
    source_files = []
    frame_ids = []

    for npz_path in progress_iter(npz_files, desc="Loading .npz files"):
        data = np.load(npz_path)
        if "poses" not in data:
            print("Skipping without poses key: {}".format(npz_path))
            continue

        poses = data["poses"].astype(np.float32)
        ids = selected_frame_ids(
            len(poses), skip_edge_ratio, stride, frames_per_file, sample_mode
        )
        if len(ids) == 0:
            print("Skipping empty frame selection: {}".format(npz_path))
            continue

        category = parse_category(npz_path, category_regex)
        pose_chunks.append(poses[ids, 3:66])
        categories.extend([category] * len(ids))
        source_files.extend([npz_path] * len(ids))
        frame_ids.extend(ids.tolist())

    if not pose_chunks:
        raise ValueError("No pose frames were loaded from the selected .npz files.")

    pose_body = torch.tensor(np.concatenate(pose_chunks), dtype=torch.float32)
    return (
        pose_body,
        np.asarray(categories),
        np.asarray(source_files),
        np.asarray(frame_ids, dtype=np.int64),
    )


def encode_pose_sequence_with_progress(vp_model, pose_body, batch_size, device):
    """Encode frames independently and collect latent means with a progress bar."""
    latents = []
    batch_starts = range(0, len(pose_body), batch_size)
    total_batches = int(np.ceil(float(len(pose_body)) / float(batch_size)))

    with torch.no_grad():
        for start in progress_iter(
            batch_starts, desc="Encoding frames", total=total_batches
        ):
            batch = pose_body[start : start + batch_size].to(device)
            encoded = vp_model.encode(batch)
            q_z = encoded["latents"] if isinstance(encoded, dict) else encoded
            latents.append(q_z.mean.detach().cpu())

    return torch.cat(latents).numpy()


def write_category_counts(categories, source_files, out_csv):
    """Write per-category file and frame counts."""
    frames_by_category = Counter(categories)
    files_by_category = defaultdict(set)
    for category, source_file in zip(categories, source_files):
        files_by_category[category].add(source_file)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "num_files", "num_frames"])
        for category in sorted(frames_by_category):
            writer.writerow(
                [
                    category,
                    len(files_by_category[category]),
                    frames_by_category[category],
                ]
            )


def category_color_map(categories):
    """Map category strings to integer color ids."""
    unique_categories = np.asarray(sorted(set(categories)))
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}
    color_ids = np.asarray([category_to_id[category] for category in categories])
    return unique_categories, color_ids


def save_category_plot(
    points_2d,
    categories,
    out_png,
    title,
    max_legend_categories,
    label_centroids,
):
    """Save a category-colored 2D latent scatter plot."""
    unique_categories, color_ids = category_color_map(categories)
    cmap_name = "tab20" if len(unique_categories) <= 20 else "nipy_spectral"
    cmap = plt.get_cmap(cmap_name, len(unique_categories))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter = ax.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=color_ids,
        s=8,
        cmap=cmap,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")

    if label_centroids:
        for idx, category in enumerate(unique_categories):
            mask = color_ids == idx
            centroid = points_2d[mask].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                category,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )

    if len(unique_categories) <= max_legend_categories:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=cmap(idx),
                markeredgecolor="none",
                markersize=6,
                label=category,
            )
            for idx, category in enumerate(unique_categories)
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )
    else:
        fig.colorbar(scatter, ax=ax, label="category id")
        ax.text(
            0.01,
            0.01,
            "Legend omitted: {} categories. See category_counts.csv.".format(
                len(unique_categories)
            ),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_category_counts_plot(categories, out_png):
    """Save a simple bar chart of sampled frame counts per category."""
    counts = Counter(categories)
    labels = sorted(counts)
    values = [counts[label] for label in labels]

    fig_height = max(4, min(18, 0.28 * len(labels)))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=150)
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("sampled frames")
    ax.set_title("Sampled frames per category")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run_projection_methods(method):
    return ["pca", "umap"] if method == "both" else [method]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root folder searched recursively for *_poses.npz files.",
    )
    parser.add_argument(
        "--expr-dir",
        default=DEFAULT_VPOSER_V1_DIR,
        help="VPoser experiment directory. Defaults to support_data/dowloads/vposer_v1_0.",
    )
    parser.add_argument("--out-dir", default="latent_category_probe_out")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--method", default="umap", choices=["pca", "umap", "both"])
    parser.add_argument(
        "--frames-per-file",
        type=int,
        default=100,
        help="Number of frames sampled from each motion file. Use -1 for all selected frames.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--skip-edge-ratio",
        type=float,
        default=0.1,
        help="Fraction skipped from the beginning and end of each sequence.",
    )
    parser.add_argument("--sample-mode", default="even", choices=["even", "random"])
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--category-regex",
        default=None,
        help="Optional regex applied to filename stem. If it has a capture group, group 1 is used.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--6DOF",
        dest="six_dof",
        type=str2bool,
        default=False,
        help="Set True to load this branch's trained 6DOF-input VPoser.",
    )
    parser.add_argument(
        "--max-legend-categories",
        type=int,
        default=25,
        help="Omit the full legend when there are more categories than this.",
    )
    parser.add_argument(
        "--label-centroids",
        action="store_true",
        help="Draw category names at their projected centroids.",
    )
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    frames_per_file = None if args.frames_per_file < 0 else args.frames_per_file
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    npz_files = find_npz_files(args.data_dir, args.max_files)
    if not npz_files:
        raise ValueError("No *_poses.npz files found under {}".format(args.data_dir))

    print("Found {} motion files.".format(len(npz_files)))
    pose_body, categories, source_files, frame_ids = load_sampled_poses(
        npz_files=npz_files,
        frames_per_file=frames_per_file,
        skip_edge_ratio=args.skip_edge_ratio,
        stride=args.stride,
        sample_mode=args.sample_mode,
        category_regex=args.category_regex,
    )
    print(
        "Loaded {} sampled frames across {} categories.".format(
            len(pose_body), len(set(categories))
        )
    )

    write_category_counts(
        categories, source_files, osp.join(args.out_dir, "category_counts.csv")
    )
    save_category_counts_plot(
        categories, osp.join(args.out_dir, "category_counts.png")
    )

    vp_model = load_vposer_model(args.expr_dir, device, args.six_dof)
    latents = encode_pose_sequence_with_progress(
        vp_model, pose_body, args.batch_size, device
    )

    projected = {}
    for method in run_projection_methods(args.method):
        print("Projecting latents with {}...".format(method))
        points_2d = project_latents(latents, method)
        projected[method] = points_2d
        out_png = osp.join(args.out_dir, "latent_categories_{}.png".format(method))
        save_category_plot(
            points_2d=points_2d,
            categories=categories,
            out_png=out_png,
            title="VPoser latent categories ({})".format(method),
            max_legend_categories=args.max_legend_categories,
            label_centroids=args.label_centroids,
        )
        print("Plot saved to {}".format(out_png))

    save_kwargs = {
        "latents": latents,
        "categories": categories,
        "source_files": source_files,
        "frame_ids": frame_ids,
        "data_dir": args.data_dir,
        "expr_dir": args.expr_dir,
        "six_dof": args.six_dof,
    }
    for method, points_2d in projected.items():
        save_kwargs["points_2d_{}".format(method)] = points_2d

    np.savez(osp.join(args.out_dir, "latent_category_probe.npz"), **save_kwargs)

    print("Saved arrays to {}".format(osp.join(args.out_dir, "latent_category_probe.npz")))
    print("Saved category counts to {}".format(osp.join(args.out_dir, "category_counts.csv")))
    print(
        "Reminder: VPoser encodes static pose frames. Categories separated mainly "
        "by speed may overlap more than categories separated by posture."
    )


if __name__ == "__main__":
    main()
