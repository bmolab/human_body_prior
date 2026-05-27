# Latent Space Evaluation Helpers

The scripts below are our newly added helpers that check the learned latent representation.

Run commands from the repository root.


## `latent_space_probe.py`

Dataset-level scatter plot for prepared VPoser datasets. It reads **one prepared
split**, encodes sampled pose frames, projects latent codes to 2D, and colors each
point by a selected **feature**.

Use this for a broad view of many independent frames from `train`, `vald`, or
`test`.

```powershell
python -m human_body_prior.evaluations.latent_space_probe `
  --expr-dir <EXPR_DIR> ` #VPoser directory 
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID> `
  --split vald `
  --6DOF True `
  --method pca ` # pca or umap
  --feature <Feature_Name> `
  --out-dir <DIR_NAME>
```

Outputs:

```text
latent_probe_<split>.npz
latent_probe_<split>_<method>.png
```

Notes:

- `--6DOF False` uses the original/pretrained VPoser by default.
- `--6DOF True` requires `--expr-dir` to point to a trained VPoser
  directory with `snapshots/*.ckpt`.
- `--model-variant` can be used to choose between supported model-loading paths.
- `pose_delta` is only a rough diagnostic feature for prepared datasets because
  the default AMASS preparation samples frames randomly.

## `latent_video_probe.py`

**Sequence-level** latent trajectory plot. It reads **one ordered AMASS `.npz` file**,
extracts `poses[:, 3:66]`, encodes the selected frames, computes a per-frame
feature, and writes trajectory plots.

Use this when you want to check things like how a feature changes through one 
sequence, and how that sequence moves through VPoser latent space.

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz <File_Path> `
  --6DOF False ` #True or False
  --method pca ` # pca or umap
  --feature <Feature_name> `
  --out-dir <DIR_NAME>
```

Useful frame-selection options:

```powershell
--start-frame <START> --end-frame <END> --stride <STRIDE>
```

Outputs:

```text
latent_video_probe.npz
feature_over_time.png
latent_trajectory_time.png
latent_trajectory_feature.png
```

Note: This can use a real per-frame torque or effort signal without
editing code:

```powershell
python -m human_body_prior.evaluations.latent_video_probe `
  --npz <File_Path>
  --6DOF False `
  --method pca `
  --feature external `
  --feature-file <File_Path> `
  --out-dir <DIR_NAME>
```

For `.npz` feature files with multiple arrays, choose the array:

```powershell
--feature-file D:\path\to\torque_features.npz --feature-key torque
```


## `latent_feature_probe.py`

**Note: This is under testing.**

Regression probe for one `latent_video_probe.py` result. It tests whether the
latent code can predict the selected per-frame feature:

```text
latent z[t] -> feature[t]
```

Run this **before training a feature-aware VPoser**. If the original latent space
already predicts the feature well, an auxiliary training target may be less
necessary.

```powershell
python -m human_body_prior.evaluations.latent_feature_probe `
  --input latent_probe\video_probe_walking_torque_proxy\latent_video_probe.npz `
  --model ridge `
  --split-mode temporal # "temporal", "random"
```

Try the nonlinear probe:

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

Important options:

- `--model ridge` checks whether the feature is linearly visible in latent
  space.
- `--model mlp` checks for a simple nonlinear relationship.
- `--split-mode temporal` trains on earlier frames and tests on later frames.
- `--split-mode random` is useful for quick checks, but can overestimate
  performance when neighboring frames are very similar.

## `latent_category_probe.py`

**Category-level** latent plot for raw AMASS-style motion folders. It recursively
finds `*_poses.npz` files, samples frames from each motion file, infers a motion
category from the filename, and **colors latent points by category**.

Use this to check whether static VPoser pose latents separate broad motion
labels such as walking, running, jumping, or lifting.

```powershell
python -m human_body_prior.evaluations.latent_category_probe `
  --data-dir <AMASS_Dataset_Dir>
  --6DOF False `
  --method umap ` #"pca", "umap", "both"
  --frames-per-file 30
  --out-dir <DIR_NAME>
```

Outputs:

```text
latent_category_probe.npz
latent_categories_<method>.png
category_counts.csv
category_counts.png
```

Useful options:

- `--frames-per-file` controls how many frames are sampled from each motion.
- `--sample-mode even` spreads samples through each sequence. 
- `--sample-mode random` samples random frames from each sequence.
- `--category-regex` overrides the default filename-based category parser.
- `--label-centroids` writes category labels near cluster centers.



## Experiments we did before training a new VPoser

1. Use `latent_video_probe.py` on one ordered AMASS/video-like sequence.
2. Use `latent_feature_probe.py` to check whether VPoser latents predict the feature.
3. Use `latent_category_probe.py` when the question is category separation across many files.
4. If the target feature is weakly represented, train a modified VPoser for that feature.
