"""Test whether VPoser latents predict a per-frame feature.

This script reads the .npz produced by latent_video_probe.py and fits a small
regression probe:

    latent z[t] -> feature[t]

Use this before training a new VPoser. If a simple probe predicts the feature
well, the original latent space already contains that information.

Input Example:
    python -m human_body_prior.evaluations.latent_feature_probe \
        --input video_probe_walking_torque_proxy/latent_video_probe.npz \
        --model ridge \
        --out-dir video_probe_walking_torque_proxy/probe
"""

import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor


def load_probe_data(input_npz):
    """Load latents/features"""
    data = np.load(input_npz, allow_pickle=True)
    if "latents" not in data or "feature" not in data:
        raise ValueError(
            "Expected 'latents' and 'feature' arrays in {}".format(input_npz)
        )

    latents = data["latents"].astype(np.float32)
    feature = data["feature"].astype(np.float32).reshape(-1)
    feature_name = str(data["feature_name"]) if "feature_name" in data else "feature"

    # Drop any frames with NaN/Inf feature or latent values before fitting.
    valid = np.isfinite(feature)
    valid = valid & np.all(np.isfinite(latents), axis=1)
    return latents[valid], feature[valid], feature_name


def make_probe_model(model_name):
    """Create a small regression probe from latent z to the selected feature.

    Ridge: main sanity check: if this works, the feature is linearly visible in the latent space.
    MLP: checks for nonlinear/entangled feature information.
    """
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "mlp":
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=1e-4,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=50,
                random_state=13,
            ),
        )
        return TransformedTargetRegressor(
            regressor=mlp,
            transformer=StandardScaler(),
        )
    raise ValueError("Unknown probe model: {}".format(model_name))


def split_data(latents, feature, test_size, random_state, split_mode):
    """Split frames for probe evaluation.

    Temporal split is the default for video analysis:
    - train on the first part of the sequence
    - test on later frames.

    Random split is only for quick checks.
    """
    if split_mode == "random":
        from sklearn.model_selection import train_test_split

        return train_test_split(
            latents,
            feature,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    if split_mode == "temporal":
        split_idx = int(round(len(feature) * (1.0 - test_size)))
        split_idx = max(1, min(split_idx, len(feature) - 1))
        return (
            latents[:split_idx],
            latents[split_idx:],
            feature[:split_idx],
            feature[split_idx:],
        )

    raise ValueError("Unknown split mode: {}".format(split_mode))


def compute_metrics(y_true, y_pred):
    """Return regression metrics in the feature's original scale."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def save_prediction_plot(y_true, y_pred, feature_name, out_png):
    """Scatter plot where perfect prediction lies on the diagonal."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_true, y_pred, s=12, alpha=0.75)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="0.3", linewidth=1)
    ax.set_title("Probe prediction: {}".format(feature_name))
    ax.set_xlabel("true {}".format(feature_name))
    ax.set_ylabel("predicted {}".format(feature_name))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_time_plot(y_true, y_pred, feature_name, out_png):
    """Plot prediction quality over frames."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(y_true, label="true", linewidth=1.5)
    ax.plot(y_pred, label="predicted", linewidth=1.2, alpha=0.8)
    ax.set_title("Probe prediction over held-out frames")
    ax.set_xlabel("held-out frame index")
    ax.set_ylabel(feature_name)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="latent_video_probe.npz file.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model", default="ridge", choices=["ridge", "mlp"])
    parser.add_argument(
        "--split-mode", default="temporal", choices=["temporal", "random"]
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    out_dir = args.out_dir or osp.join(
        osp.dirname(args.input), "probe_{}".format(args.model)
    )
    os.makedirs(out_dir, exist_ok=True)

    latents, feature, feature_name = load_probe_data(args.input)
    x_train, x_test, y_train, y_test = split_data(
        latents, feature, args.test_size, args.random_state, args.split_mode
    )

    probe = make_probe_model(args.model)
    probe.fit(x_train, y_train)
    y_pred = probe.predict(x_test)

    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    y_base = baseline.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)
    baseline_metrics = compute_metrics(y_test, y_base)
    metrics.update(
        {
            "input": args.input,
            "model": args.model,
            "split_mode": args.split_mode,
            "feature_name": feature_name,
            "num_train": int(len(x_train)),
            "num_test": int(len(x_test)),
            "baseline_mean": baseline_metrics,
        }
    )

    with open(osp.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_prediction_plot(
        y_test, y_pred, feature_name, osp.join(out_dir, "prediction_scatter.png")
    )
    save_time_plot(
        y_test, y_pred, feature_name, osp.join(out_dir, "prediction_heldout.png")
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
