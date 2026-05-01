# Latent Space Evaluation Helpers

For exploring how VPoser latent codes relate to hand-built or
future physics features. They are meant for small experiments on one dataset
split or one ordered motion sequence, not for full AMASS benchmarking.

## Scripts

### `latent_space_probe.py`

**Dataset-level** scatter plot. It reads a prepared VPoser dataset folder containing
`train/`, `vald/`, or `test/` `.pt` files, encodes each frame, projects latent
codes to 2D, and colors points by a feature.

Use this when you want a broad view of many independent frames.

```powershell
python -m human_body_prior.evaluations.latent_space_probe `
  --dataset-dir D:\Git\human_body_prior\AMASS\DataSet\SFU_test `
  --split vald `
  --6DOF False `
  --method pca `
  --feature pose_magnitude `
  --out-dir latent_probe_v1
```

Outputs:

```text
latent_probe_<split>.npz
latent_probe_<split>_<method>.png
```

Notes:

- `--6DOF False` uses the original/pretrained VPoser by default.
- `--6DOF True` requires `--expr-dir` to point to a trained 6DOF experiment
  directory with `snapshots/*.ckpt`.
- `pose_delta` is only a rough smoke-test feature for prepared datasets because
  the default AMASS preparation samples frames randomly.

### `latent_video_probe.py`

**Sequence-level** latent trajectory plot. It reads **one** ordered AMASS `.npz`,
extracts `poses[:, 3:66]`, encodes every frame, computes a per-frame feature,
and writes three plots:

```text
feature_over_time.png
latent_trajectory_time.png
latent_trajectory_feature.png
```

Use this for the current video-style goal: see how a feature changes through one
motion and how those changes appear in VPoser latent space.

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz D:\Git\human_body_prior\AMASS\SFU\0005\0005_Walking001_poses.npz `
  --6DOF False `
  --method pca `
  --feature torque_proxy `
  --out-dir latent_probe\video_probe_walking_torque_proxy
```

Current features live in `pose_sequence_features.py`:

```text
pose_speed
pose_acceleration
pose_magnitude
torque_proxy
```

**Note**: `torque_proxy` is not physical torque. It is an exploratory effort-like feature:

```text
pose_magnitude * (pose_speed + pose_acceleration)
```

**When real torque is available, add a new feature function in
`pose_sequence_features.py` and register it in `FEATURES`.**

You can also provide a real per-frame torque or effort signal directly without
editing code:

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz D:\Git\human_body_prior\AMASS\SFU\0005\0005_Walking001_poses.npz `
  --6DOF False `
  --method pca `
  --feature external `
  --feature-file D:\path\to\torque.npy `
  --out-dir latent_probe\video_probe_real_torque
```

For `.npz` feature files with multiple arrays, choose the array:

```powershell
--feature-file D:\path\to\torque_features.npz --feature-key torque
```

External feature shape handling:

```text
[T]       used directly as one scalar per frame
[T, J]    reduced by mean absolute value over joints
[T, J, D] reduced by vector norm, then mean over joints
```

The external feature length must match the selected sequence length after
`--start-frame`, `--end-frame`, and `--stride`.

### `latent_feature_probe.py`

Probe model. It reads `latent_video_probe.npz` and tests whether latent codes can
predict the selected feature:

```text
latent z[t] -> feature[t]
```

Run this before training a new torque-aware VPoser. It tells you whether the
original latent space already contains the feature.

```powershell
python -m human_body_prior.evaluations.latent_feature_probe `
  --input latent_probe\video_probe_walking_torque_proxy\latent_video_probe.npz `
  --model ridge `
  --split-mode temporal
```

Try the nonlinear probe (MLP):

```powershell
python -m human_body_prior.evaluations.latent_feature_probe `
  --input latent_probe\video_probe_walking_torque_proxy\latent_video_probe.npz `
  --model mlp `
  --split-mode temporal
```

Outputs:

```text
metrics.json
prediction_scatter.png
prediction_heldout.png
```

## Interpreting Probe Metrics

`r2`:

- Near `1.0`: latent codes predict the feature well.
- Around `0.0`: no better than predicting an average value.
- Below `0.0`: worse than the average predictor.

`baseline_mean`:

This is the metric for a trivial model that always predicts the training-set
mean. The probe should beat this baseline to be meaningful.

`ridge` vs `mlp`:

- Good ridge score: the feature is linearly visible in latent space.
- Low ridge but better MLP: the feature may be present but nonlinear/entangled.
- Low both: the feature is probably not strongly represented in original VPoser.

`--split-mode temporal`:

Recommended for video-style analysis. It trains on the first part of a sequence
and tests on later frames.

`--split-mode random`:

Useful for quick checks, but can overestimate performance because adjacent video
frames are often very similar.

## Experiment Workflow

1. Use `latent_video_probe.py` on one ordered AMASS/video-like sequence.
2. Compare feature plots.
3. Use `latent_feature_probe.py` to check if original VPoser latents predict the feature.
4. If the targer feature is weakly predicted from original latents, train a modified VPoser for that feature
