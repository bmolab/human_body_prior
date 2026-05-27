# Latent Space Evaluation Helpers

These helpers inspect learned VPoser latent representations. Run commands from the repository root.


## `check_vposer_dataset.py`

Quick sanity check for prepared `.pt` datasets:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS_Processed\DataSet\<PREPARED_DATASET_ID>
```

By default this checks `pose_body` and `root_orient` for `train`, `vald`, and `test`. For torque-proxy datasets, include `torque_proxy` with `--fields`.


## `latent_space_probe.py`

**Dataset-level** scatter plot for prepared VPoser datasets. It reads one prepared split, encodes sampled pose frames, projects latent codes to 2D, and colors each point by a selected feature.

Use this for a broad view of independent frames from `train`, `vald`, or `test`.

```powershell
python -m human_body_prior.evaluations.latent_space_probe `
  --expr-dir <EXPR_DIR> `
  --dataset-dir AMASS_Processed\DataSet\<PREPARED_DATASET_ID> `
  --split vald `
  --6DOF True `
  --method pca `
  --feature torque_proxy `
  --out-dir <DIR_NAME>
```


Notes:

- `--method` can be `pca` or `umap`.
- `--feature` can be `pose_magnitude`, `pose_delta`, or a prepared dataset field such as `torque_proxy`.
- `pose_delta` is only a rough diagnostic for prepared datasets because default AMASS preparation samples frames randomly.
- `--split` currently supports `train`, `vald`, and `test`.
- `--6DOF True` requires `--expr-dir` to point to a trained experiment directory with `snapshots/*.ckpt`.
- `--model-variant` choices are `v1` for packaged/pretrained VPoser, `old` for an original axis-angle branch checkpoint, and `6d` for this branch's 6D-input VPoser. `--6DOF True` automatically uses `6d`.


## `latent_video_probe.py`

**Sequence-level** latent trajectory plot. It reads one ordered AMASS `.npz` file, extracts `poses[:, 3:66]`, encodes selected frames, computes a per-frame feature, and writes **trajectory** plots.

Use this when you want to inspect how a feature changes through one sequence and how that sequence moves through VPoser latent space.

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz <FILE_PATH> `
  --6DOF False `
  --method pca `
  --feature torque_proxy `
  --out-dir <DIR_NAME>
```

Useful frame-selection options:

```powershell
--start-frame <START> --end-frame <END> --stride <STRIDE>
```


Supported built-in features are `pose_speed`, `pose_acceleration`, `pose_magnitude`, and `torque_proxy`.

To use a real per-frame torque or effort signal, pass an external `.npy` or `.npz` feature file:

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz <FILE_PATH> `
  --6DOF False `
  --method pca `
  --feature external `
  --feature-file <FEATURE_FILE> `
  --out-dir <DIR_NAME>
```

For `.npz` feature files with multiple arrays, choose the array:

```powershell
--feature-file D:\path\to\torque_features.npz --feature-key torque
```

External feature length must match the selected frames after `--start-frame`, `--end-frame`, and `--stride`.


## `latent_feature_probe.py`

Regression probe for one `latent_video_probe.py` result. It tests whether latent codes can predict the selected **per-frame feature** `latent z[t] -> feature[t]`

you can run this before training a feature-aware VPoser. If the original latent space already predicts the feature well, an auxiliary target may be less necessary.

```powershell
python -m human_body_prior.evaluations.latent_feature_probe `
  --input latent_probe\video_probe_walking_torque_proxy\latent_video_probe.npz `
  --model ridge `
  --split-mode temporal
```

nonlinear probe:

```powershell
python -m human_body_prior.evaluations.latent_feature_probe `
  --input latent_probe\video_probe_walking_torque_proxy\latent_video_probe.npz `
  --model mlp `
  --split-mode temporal
```

Important options:

- `--model ridge` checks whether the feature is linearly visible in latent space.
- `--model mlp` checks for a simple nonlinear relationship.
- `--split-mode temporal` trains on earlier frames and tests on later frames.
- `--split-mode random` is useful for quick checks, but can overestimate performance when neighboring frames are very similar.


## `latent_category_probe.py`

**Category-level** latent plot for raw AMASS-style motion folders. It recursively finds `*_poses.npz` files, samples frames from each motion file, infers a motion category from the filename, and colors latent points by category.

Use this to check whether static VPoser pose latents separate broad motion labels such as walking, running, jumping, or lifting.

```powershell
python -m human_body_prior.evaluations.latent_category_probe `
  --data-dir <AMASS_DATASET_DIR> `
  --6DOF False `
  --method umap `
  --frames-per-file 30 `
  --out-dir <DIR_NAME>
```


Useful options:

- `--method` can be `pca`, `umap`, or `both`.
- `--frames-per-file` controls how many frames are sampled from each motion.
- `--sample-mode even` spreads samples through each sequence.
- `--sample-mode random` samples random frames from each sequence.
- `--category-regex` overrides the default filename-based category parser.
- `--label-centroids` writes category labels near cluster centers.
