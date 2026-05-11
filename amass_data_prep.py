"""Prepare AMASS .npz files into the .pt files used by VPoser training.

Example:
    python amass_data_prep.py --config configs/data/amass_vposer_example.yaml
"""

import sys

import argparse
import glob
import os
import os.path as osp
import shutil
  
import yaml

REPO_ROOT = osp.dirname(osp.abspath(__file__))
SRC_DIR = osp.join(REPO_ROOT, "src")
if osp.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class SplitConfig(dict):
    def toDict(self):
        return dict(self)


def load_prep_config(config_path):
    """Load and validate the YAML config used for AMASS preprocessing."""

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError("Config file is empty: {}".format(config_path))

    required_keys = ["amass_dir", "out_dir", "splits"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            "Config file {} missing required key(s): {}".format(
                config_path, ", ".join(missing_keys)
            )
        )

    splits = config["splits"]
    for split_name in ["train", "vald", "test"]:
        if split_name not in splits:
            raise ValueError(
                "Config file {} missing splits.{}".format(config_path, split_name)
            )
        if not isinstance(splits[split_name], list) or len(splits[split_name]) == 0:
            raise ValueError(
                "splits.{} must be a non-empty list of AMASS dataset names.".format(
                    split_name
                )
            )

    return config


def split_names_from_config(splits):
    """Keep the standard training splits first, then include extra eval splits."""
    ordered_split_names = [
        split_name for split_name in ["train", "vald", "test"] if split_name in splits
    ]
    ordered_split_names.extend(
        split_name for split_name in splits.keys() if split_name not in ordered_split_names
    )
    return ordered_split_names


# For validation
def print_dataset_summary(out_dir, splits):
    """Print final prepared .pt tensor counts and shapes."""
    import torch

    split_counts = {}
    split_shapes = {}
    split_names = split_names_from_config(splits)
    for split_name in split_names:
        pose_body_fname = osp.join(out_dir, split_name, "pose_body.pt")
        if not osp.exists(pose_body_fname):
            continue

        pose_body = torch.load(pose_body_fname)
        split_counts[split_name] = int(pose_body.shape[0])
        split_shapes[split_name] = {"pose_body": tuple(pose_body.shape)}

        for field_name in ["root_orient", "torque_proxy", "source_dataset", "source_sequence"]:
            field_fname = osp.join(out_dir, split_name, "{}.pt".format(field_name))
            if osp.exists(field_fname):
                split_shapes[split_name][field_name] = tuple(
                    torch.load(field_fname).shape
                )

    total_count = sum(split_counts.values())
    print("\nPrepared dataset summary:", out_dir)
    for split_name in split_names:
        ds_names = splits.get(split_name, [])
        count = split_counts.get(split_name, 0)
        pct = (100.0 * count / total_count) if total_count else 0.0
        print(
            "  {}: {:,} frames ({:.1f}%) from AMASS subset(s): {}".format(
                split_name,
                count,
                pct,
                ", ".join(ds_names),
            )
        )
        for field_name, shape in split_shapes.get(split_name, {}).items():
            print("    {} shape: {}".format(field_name, shape))


def summarize_balanced_sampling(amass_dir, splits, preprocessing, include_torque_proxy):
    """Estimate how many valid frames each AMASS subset contributes."""
    import numpy as np

    from human_body_prior.data.prepare_data import _compute_torque_proxy
    from human_body_prior.data.prepare_data import build_frame_balanced_sequence_assignments

    keep_rate = preprocessing.get("keep_rate", 0.3)
    max_frames_per_sequence = preprocessing.get("max_frames_per_sequence")
    max_frames_per_dataset = preprocessing.get("max_frames_per_dataset")
    torque_proxy_use_fps = preprocessing.get("torque_proxy_use_fps", True)
    torque_proxy_filter_percentile = preprocessing.get(
        "torque_proxy_filter_percentile", 99.5
    )
    split_sequences_by_source = preprocessing.get("split_sequences_by_source", False)
    sequence_split_ratios = preprocessing.get(
        "sequence_split_ratios", {"train": 0.8, "vald": 0.1, "test": 0.1}
    )
    sequence_split_seed = preprocessing.get("sequence_split_seed", 100)

    def finite_pose_mask(poses):
        """Keep frames whose body-pose values are finite."""
        return np.isfinite(poses[:, :66]).all(axis=1)

    def sample_count(candidate_count, remaining_dataset_budget=None):
        """Match the preprocessing sampler without loading final tensors."""
        if candidate_count < 1:
            return 0
        count = max(1, int(keep_rate * candidate_count))
        if max_frames_per_sequence is not None:
            count = min(count, int(max_frames_per_sequence))
        if remaining_dataset_budget is not None:
            count = min(count, int(remaining_dataset_budget))
        return max(0, min(count, candidate_count))

    sequence_assignments = build_frame_balanced_sequence_assignments(
        amass_dir,
        splits,
        split_sequences_by_source=split_sequences_by_source,
        sequence_split_ratios=sequence_split_ratios,
        sequence_split_seed=sequence_split_seed,
        keep_rate=keep_rate,
        max_frames_per_sequence=max_frames_per_sequence,
        include_torque_proxy=include_torque_proxy,
        torque_proxy_use_fps=torque_proxy_use_fps,
        torque_proxy_filter_percentile=torque_proxy_filter_percentile,
    )

    summary = {}
    for split_name in split_names_from_config(splits):
        split_summary = {"frames_selected": 0, "datasets": {}}
        for ds_name in splits.get(split_name, []):
            npz_fnames = sequence_assignments.get((split_name, ds_name), [])
            ds_summary = {
                "sequences_found": len(npz_fnames),
                "sequences_used": 0,
                "frames_seen": 0,
                "candidate_frames": 0,
                "valid_candidate_frames": 0,
                "filtered_candidate_frames": 0,
                "frames_selected": 0,
                "max_frames_per_dataset_reached": False,
            }

            for npz_fname in sorted(npz_fnames):
                if max_frames_per_dataset is not None and ds_summary[
                    "frames_selected"
                ] >= int(max_frames_per_dataset):
                    ds_summary["max_frames_per_dataset_reached"] = True
                    break

                with np.load(npz_fname) as cdata:
                    poses = cdata["poses"].astype(np.float32)
                    n_frames = len(poses)
                    candidate_ids = np.asarray(
                        list(range(int(0.1 * n_frames), int(0.9 * n_frames), 1)),
                        dtype=np.int64,
                    )

                    ds_summary["frames_seen"] += int(n_frames)
                    ds_summary["candidate_frames"] += int(len(candidate_ids))
                    if len(candidate_ids) < 1:
                        continue

                    valid_mask = finite_pose_mask(poses)
                    if include_torque_proxy:
                        mocap_framerate = None
                        if torque_proxy_use_fps and "mocap_framerate" in cdata.files:
                            mocap_framerate = cdata["mocap_framerate"]
                        torque_proxy, pose_speed, pose_acceleration = (
                            _compute_torque_proxy(
                                poses,
                                mocap_framerate=mocap_framerate,
                                return_components=True,
                            )
                        )
                        valid_mask &= np.isfinite(torque_proxy)
                        valid_mask &= np.isfinite(pose_speed)
                        valid_mask &= np.isfinite(pose_acceleration)

                        finite_proxy = torque_proxy[np.isfinite(torque_proxy)]
                        if (
                            torque_proxy_filter_percentile is not None
                            and len(finite_proxy) > 0
                        ):
                            proxy_limit = np.percentile(
                                finite_proxy,
                                float(torque_proxy_filter_percentile),
                            )
                            valid_mask &= torque_proxy <= proxy_limit

                    valid_candidate_count = int(valid_mask[candidate_ids].sum())
                    ds_summary["valid_candidate_frames"] += valid_candidate_count

                    remaining_dataset_budget = None
                    if max_frames_per_dataset is not None:
                        remaining_dataset_budget = (
                            int(max_frames_per_dataset) - ds_summary["frames_selected"]
                        )
                    chosen_count = sample_count(
                        valid_candidate_count, remaining_dataset_budget
                    )
                    if chosen_count < 1:
                        continue

                    ds_summary["sequences_used"] += 1
                    ds_summary["frames_selected"] += chosen_count

            ds_summary["filtered_candidate_frames"] = (
                ds_summary["candidate_frames"] - ds_summary["valid_candidate_frames"]
            )
            split_summary["frames_selected"] += ds_summary["frames_selected"]
            split_summary["datasets"][ds_name] = ds_summary
        summary[split_name] = split_summary

    return summary


def print_balanced_sampling_summary(summary):
    """Print the per-split and per-source frame summary."""
    
    total_frames = sum(split["frames_selected"] for split in summary.values())
    print("\nBalanced sampling summary from AMASS source files:")
    for split_name in split_names_from_config(summary):
        split_summary = summary.get(split_name, {"frames_selected": 0, "datasets": {}})
        split_frames = split_summary["frames_selected"]
        split_pct = (100.0 * split_frames / total_frames) if total_frames else 0.0
        print(
            "  {}: {:,} planned frames ({:.1f}%)".format(
                split_name, split_frames, split_pct
            )
        )
        for ds_name, ds_summary in split_summary["datasets"].items():
            cap_note = (
                " cap-hit" if ds_summary["max_frames_per_dataset_reached"] else ""
            )
            print(
                "    {}: {:,} frames from {}/{} sequences; valid {:,}/{:,}, filtered {:,}{}".format(
                    ds_name,
                    ds_summary["frames_selected"],
                    ds_summary["sequences_used"],
                    ds_summary["sequences_found"],
                    ds_summary["valid_candidate_frames"],
                    ds_summary["candidate_frames"],
                    ds_summary["filtered_candidate_frames"],
                    cap_note,
                )
            )


def clean_existing_splits(out_dir, splits):
    # Ask before deleting existing prepared split folders.
    split_dirs = [
        osp.join(out_dir, split_name)
        for split_name in split_names_from_config(splits)
    ]
    existing_split_dirs = [
        split_dir for split_dir in split_dirs if osp.exists(split_dir)
    ]
    if not existing_split_dirs:
        return

    print("\nExisting prepared split folder(s) found:")
    for split_dir in existing_split_dirs:
        print("  {}".format(split_dir))

    answer = (
        input("Clean these existing split folder(s) before preprocessing? [y/N]: ")
        .strip()
        .lower()
    )
    if answer not in ("y", "yes"):
        print("Keeping existing prepared split folder(s).")
        return

    for split_dir in existing_split_dirs:
        if osp.isdir(split_dir):
            shutil.rmtree(split_dir)
        else:
            os.remove(split_dir)
    print("Cleaned existing prepared split folder(s).")


def main():
    """Run AMASS preprocessing and print dataset summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=osp.join("configs", "data", "amass_vposer_example.yaml"),
        help="YAML config describing AMASS input, output, and train/vald/test splits.",
    )
    args = parser.parse_args()

    config = load_prep_config(args.config)
    amass_dir = config["amass_dir"]
    out_dir = config["out_dir"]
    amass_splits = SplitConfig(config["splits"])

    from human_body_prior.data.prepare_data import prepare_vposer_datasets
    from human_body_prior.tools.omni_tools import log2file, makepath

    log_name = config.get("log_name", "prepare.log")
    logger = log2file(makepath(out_dir, log_name, isfile=True))
    logger("Preparing VPoser dataset using config: {}".format(args.config))
    logger("AMASS input directory: {}".format(amass_dir))
    logger("Prepared dataset output directory: {}".format(out_dir))
    logger("AMASS splits: {}".format(config["splits"]))

    clean_existing_splits(out_dir, config["splits"])

    include_torque_proxy = bool(config.get("include_torque_proxy", False))
    preprocessing = config.get("preprocessing", {})
    prepare_vposer_datasets(
        out_dir,
        amass_splits,
        amass_dir,
        logger=logger,
        include_torque_proxy=include_torque_proxy,
        keep_rate=preprocessing.get("keep_rate", 0.3),
        max_frames_per_sequence=preprocessing.get("max_frames_per_sequence"),
        max_frames_per_dataset=preprocessing.get("max_frames_per_dataset"),
        torque_proxy_use_fps=preprocessing.get("torque_proxy_use_fps", True),
        torque_proxy_log1p=preprocessing.get("torque_proxy_log1p", True),
        torque_proxy_filter_percentile=preprocessing.get(
            "torque_proxy_filter_percentile", 99.5
        ),
        save_source_metadata=preprocessing.get("save_source_metadata", True),
        split_sequences_by_source=preprocessing.get(
            "split_sequences_by_source", False
        ),
        sequence_split_ratios=preprocessing.get("sequence_split_ratios"),
        sequence_split_seed=preprocessing.get("sequence_split_seed", 100),
    )
    print("Prepared dataset at:", out_dir)
    print_balanced_sampling_summary(
        summarize_balanced_sampling(
            amass_dir,
            config["splits"],
            preprocessing,
            include_torque_proxy,
        )
    )
    print_dataset_summary(out_dir, config["splits"])


if __name__ == "__main__":
    main()
