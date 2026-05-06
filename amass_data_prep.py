"""Prepare AMASS .npz files into the .pt files used by VPoser training.

Example:
    python amass_data_prep.py --config configs/data/amass_vposer_example.yaml
"""

import argparse
import os
import os.path as osp
import shutil
import sys

REPO_ROOT = osp.dirname(osp.abspath(__file__))
SRC_DIR = osp.join(REPO_ROOT, "src")
if osp.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class SplitConfig(dict):
    def toDict(self):
        return dict(self)


def load_prep_config(config_path):
    import yaml

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError("Config file is empty: {}".format(config_path))

    required_keys = ["amass_dir", "out_dir", "splits"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            "Config file {} is missing required key(s): {}".format(
                config_path, ", ".join(missing_keys)
            )
        )

    splits = config["splits"]
    for split_name in ["train", "vald", "test"]:
        if split_name not in splits:
            raise ValueError(
                "Config file {} is missing splits.{}".format(config_path, split_name)
            )
        if not isinstance(splits[split_name], list) or len(splits[split_name]) == 0:
            raise ValueError(
                "splits.{} must be a non-empty list of AMASS dataset names.".format(
                    split_name
                )
            )

    return config


def print_dataset_summary(out_dir, splits):
    import torch

    split_counts = {}
    split_shapes = {}
    for split_name in ["train", "vald", "test"]:
        pose_body_fname = osp.join(out_dir, split_name, "pose_body.pt")
        if not osp.exists(pose_body_fname):
            continue

        pose_body = torch.load(pose_body_fname)
        split_counts[split_name] = int(pose_body.shape[0])
        split_shapes[split_name] = {"pose_body": tuple(pose_body.shape)}

        for field_name in ["root_orient", "torque_proxy"]:
            field_fname = osp.join(out_dir, split_name, "{}.pt".format(field_name))
            if osp.exists(field_fname):
                split_shapes[split_name][field_name] = tuple(torch.load(field_fname).shape)

    total_count = sum(split_counts.values())
    print("\nPrepared dataset summary:", out_dir)
    for split_name in ["train", "vald", "test"]:
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


def maybe_clean_existing_splits(out_dir):
    split_dirs = [osp.join(out_dir, split_name) for split_name in ["train", "vald", "test"]]
    existing_split_dirs = [split_dir for split_dir in split_dirs if osp.exists(split_dir)]
    if not existing_split_dirs:
        return

    print("\nExisting prepared split folder(s) found:")
    for split_dir in existing_split_dirs:
        print("  {}".format(split_dir))

    answer = input("Clean these existing split folder(s) before preprocessing? [y/N]: ").strip().lower()
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

    maybe_clean_existing_splits(out_dir)

    include_torque_proxy = bool(config.get("include_torque_proxy", False))
    prepare_vposer_datasets(
        out_dir,
        amass_splits,
        amass_dir,
        logger=logger,
        include_torque_proxy=include_torque_proxy,
    )
    print("Prepared dataset at:", out_dir)
    print_dataset_summary(out_dir, config["splits"])


if __name__ == "__main__":
    main()
