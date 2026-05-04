"""Prepare AMASS .npz files into the .pt files used by VPoser training.

Example:
    python amass_data_prep.py --config configs/data/amass_vposer_example.yaml
"""

import argparse
import os.path as osp
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

    prepare_vposer_datasets(out_dir, amass_splits, amass_dir, logger=logger)
    print("Prepared dataset at:", out_dir)


if __name__ == "__main__":
    main()
