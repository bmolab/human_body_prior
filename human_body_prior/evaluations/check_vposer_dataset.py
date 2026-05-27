"""

This script is for debugging a common issue: When you see NaN value error during training.
 
It Check prepared VPoser .pt datasets for non-finite pose values.

Example:
    python -m human_body_prior.evaluations.check_vposer_dataset \
        --dataset-dir human_body_prior\AMASS\DataSet\SFU_test
"""

import argparse
import os.path as osp

import torch


def check_tensor(path):
    tensor = torch.load(path).float()
    finite = torch.isfinite(tensor)
    result = {
        "path": path,
        "shape": list(tensor.shape),
        "finite": bool(finite.all().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
    }
    if finite.any():
        finite_values = tensor[finite]
        result["min"] = float(finite_values.min().item())
        result["max"] = float(finite_values.max().item())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "vald", "test"])
    parser.add_argument("--fields", nargs="+", default=["pose_body", "root_orient"])
    args = parser.parse_args()

    has_problem = False
    for split in args.splits:
        for field in args.fields:
            path = osp.join(args.dataset_dir, split, "{}.pt".format(field))
            result = check_tensor(path)
            has_problem = has_problem or not result["finite"]
            print(result)

    if has_problem:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
