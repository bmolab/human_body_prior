"""Helpers for recording and plotting VPoser training losses."""

import argparse
import csv
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime


LOSS_HISTORY_FNAME = "loss_history.csv"
LOSS_PLOT_FNAME = "loss_history.png"


class LossHistoryRecorder:
    """Collect per-batch loss values and write epoch means to CSV."""

    fieldnames = ["time", "epoch", "split", "metric", "value"]

    def __init__(self, work_dir, logger=None):
        self.work_dir = work_dir
        self.csv_path = osp.join(work_dir, LOSS_HISTORY_FNAME)
        self.logger = logger
        self._values = defaultdict(lambda: defaultdict(list))

    def reset(self, split=None):
        if split is None:
            self._values.clear()
        else:
            self._values.pop(split, None)

    def update(self, split, loss_dict):
        for metric, value in loss_dict.items():
            self._values[split][metric].append(_to_float(value))

    def write_epoch(self, split, epoch):
        split_values = self._values.get(split, {})
        if not split_values:
            return

        os.makedirs(self.work_dir, exist_ok=True)
        needs_header = not osp.exists(self.csv_path) or osp.getsize(self.csv_path) == 0
        timestamp = datetime.now().isoformat(timespec="seconds")

        with open(self.csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            if needs_header:
                writer.writeheader()
            for metric, values in sorted(split_values.items()):
                writer.writerow(
                    {
                        "time": timestamp,
                        "epoch": int(epoch),
                        "split": split,
                        "metric": metric,
                        "value": sum(values) / len(values),
                    }
                )

        if self.logger is not None:
            self.logger("Wrote {} loss history to {}".format(split, self.csv_path))


def _to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def read_loss_history(csv_path):
    series = defaultdict(lambda: {"epochs": [], "values": []})
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            key = "{} {}".format(row["split"], row["metric"])
            series[key]["epochs"].append(int(row["epoch"]))
            series[key]["values"].append(float(row["value"]))
    return series


def plot_loss_history(csv_path, out_path=None):
    if out_path is None:
        out_path = osp.join(osp.dirname(csv_path), LOSS_PLOT_FNAME)

    series = read_loss_history(csv_path)
    if not series:
        raise ValueError("No loss rows found in {}".format(csv_path))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6), dpi=150)
    for label, data in sorted(series.items()):
        plt.plot(data["epochs"], data["values"], marker="o", linewidth=1.5, label=label)

    plt.xlabel("epoch")
    plt.ylabel("weighted loss")
    plt.title("VPoser training loss history")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot VPoser loss_history.csv.")
    parser.add_argument("csv_path", help="Path to loss_history.csv")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()
    out_path = plot_loss_history(args.csv_path, args.out)
    print("Loss plot saved to {}".format(out_path))


if __name__ == "__main__":
    main()
