# Modified VPoser Training Workflow

This note covers the current end-to-end workflow for the modified VPoser training pipeline. It focuses on the framework changes and the commands needed to run the pipeline, not on covering every experiment variant. For more development-related details (e.g. what we experimented with), please check [DevNotes](https://github.com/bmolab/knowledge-share-docs/blob/main/human_body_prior/Dev_Notes_202605.md).

High-level changes:
- The main VPoser implementation now encodes poses through a continuous 6D rotation representation.
- AMASS preprocessing was extended to prepare balanced multi-source datasets, optional held-out cross-dataset splits, optional source metadata, and an exploratory torque-proxy target.
- Torque-proxy training adds an auxiliary effort-like prediction target to VPoser training.
- Evaluation helpers were added to inspect learned latent spaces and check whether pose or effort-like features are visible in the representation.


## 1. Install

Follow the main [`README.md`](README.md).

If you get an error saying it cannot find `human_body_prior`, run: `python setup.py develop`


## 2. Prepare Raw AMASS

Put raw AMASS motion files under:

```text
AMASS\<DATASET_NAME>\...\*_poses.npz
```

The preprocessing reads:

```text
root_orient = poses[:, :3]
pose_body   = poses[:, 3:66]
```


## 3. Preprocess AMASS

Training does not read raw AMASS `.npz` files directly now. So you need to start with **setting up a config in `configs\data\`**. A sample config with detailed comments is available at [`configs\data\amass_vposer_in_domain_cross_sample.yaml`](configs/data/amass_vposer_in_domain_cross_sample.yaml).


The important settings in the config file are:
- Prepared datasets are written under the `out_dir`
- `splits`: split datasets into `train`, `vald`, `test`, and optional held-out splits such as `vald_cross` and `test_cross`. 
  - If the same AMASS source appears under `train`, `vald`, and `test`, enable `split_sequences_by_source` so entire source sequences are assigned to one in-domain split. This avoids neighboring frames from the same sequence leaking across train/validation/test.
  - `vald_cross` and `test_cross` can be prepared for held-out dataset checks, but the trainer uses `vald` for early stopping and checkpoint selection.
- `include_torque_proxy`: writes `torque_proxy.pt` for torque-proxy training.
- `preprocessing`: optional, it keeps torque-proxy training more stable across AMASS sources. It handles outliers (filters non-finite/extreme frames), compresses long-tailed values, and caps frames per sequence/dataset so very large subsets do not dominate.



**Run preprocessing**:

```powershell
python amass_data_prep.py --config configs\data\amass_vposer_in_domain_cross_sample.yaml
```

**Validate the prepared dataset**:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS_Processed\DataSet\<PREPARED_DATASET_ID>
```

Expected result:

- all `finite` is `True`
- `nan_count` is `0`
- `inf_count` is `0`

By default, this check validates `pose_body` and `root_orient`. For torque-proxy datasets, you can also include `torque_proxy` with the `--fields` option.

**Note**:
The prepared datasets also include:

```text
train\source_dataset.pt
train\source_sequence.pt
source_dataset_names.json
source_sequence_names.json
```

Training uses this metadata to log **validation metrics per source dataset** (e.g. `vald/CMU/loss_total` and `vald/KIT/loss_total`) when it is present.


## 4. Training

You need:

- a prepared dataset under `AMASS_Processed\DataSet\<PREPARED_DATASET_ID>`
- a SMPL/SMPL-X body model `.npz` file

Training experiment entry points live under:

```text
human_body_prior\train\<EXPERIMENT_FOLDER>
```

Each experiment folder contains:

```text
<EXPERIMENT_NAME>.py  # the command-line entry point
<EXPERIMENT_NAME>.yaml  # stores the default model, data, and training settings.
```

Run torque-proxy training with a dataset prepared using `include_torque_proxy: true`:

```powershell
python -m human_body_prior.train.V02_08_6D_torque_proxy.V02_08_6D_torque_proxy `
  --dataset-dir AMASS_Processed\DataSet\<PREPARED_DATASET_ID> `
  --expr-id <EXPR_ID> `
  --bm-fname <BODY_MODEL_PATH>
```

Outputs are written under:

```text
support_data\training\training_experiments\<EXPR_ID>
```

Useful outputs:

```text
loss_history.csv
snapshots\*.ckpt
tensorboard\version_*
<EXPR_ID>.log
```

Monitor TensorBoard:

```powershell
tensorboard --logdir support_data\training\training_experiments\<EXPR_ID>\tensorboard
```

`val_loss` on the `vald` split is used for early stopping and checkpointing. The selected checkpoint path is written to the training log as `best_model_fname`.


## 5. Framework Changes Compared to Original VPoser

The original VPoser encoder reads axis-angle pose:

```text
21 joints * 3 axis-angle values = 63 values
```

Now the main `VPoser` converts each joint rotation to a continuous 6D representation before encoding:

```text
21 joints * 6D rotation values = 126 values
```

The decoder still reconstructs body pose through rotation matrices and returns axis-angle pose for the body model. `VPoserOld` remains available for loading the original/pretrained axis-angle VPoser path.

We implemented a few models at a high level:

- `VPoser`: the main branch model, using 6D rotation input internally.
- `VPoserTorqueProxy`: keeps the same pose encoder/decoder and adds an auxiliary torque-proxy prediction head from the latent representation.
- `VPoserDecoderTorqueProxy`: keeps pose reconstruction as the main task and predicts the torque proxy from shared decoder-side features.

**Note**: Torque-proxy training adds an auxiliary prediction target to VPoser training. The target is computed during preprocessing from ordered AMASS motion sequences **before random frame sampling**. Currently we use pose magnitude, pose speed, and pose acceleration as an exploratory effort-like proxy. It is not physical inverse-dynamics torque.

The baseline VPoser-style training objective uses KL regularization plus pose reconstruction terms:

```text
loss_total = loss_kl + matrot + jtr
```

For torque-proxy training, wee add a weighted auxiliary proxy loss:

```text
loss_total = loss_kl + matrot + jtr + torque_proxy
```

Terms:

- `loss_kl`: keeps the latent distribution close to `N(0, I)`.
- `matrot`: compares reconstructed joint rotations with original rotations.
- `jtr`: compares reconstructed body joints with original body joints after SMPL-X forward kinematics.
- `torque_proxy`: compares the predicted scalar with the preprocessed exploratory torque-proxy target.

Notes:

- `matrot` and `jtr` are included while `current_epoch < keep_extra_loss_terms_until_epoch`.
- `loss_rec_wt` remains in the config but we are not actually using it. 

- The proxy loss compares the model's predicted scalar with the preprocessed `torque_proxy.pt` target. The trainer supports proxy normalization and robust loss options through config. See `human_body_prior\train\vposer_trainer.py` for the implementation details.


## 6. Evaluation Probes

The evaluation helpers live under:

```text
human_body_prior\evaluations
```

Use them after training to inspect latent spaces, sequence trajectories, and feature visibility. See [`human_body_prior\evaluations\README.md`](human_body_prior/evaluations/README.md) for the commands.


## Troubleshooting

If training gives NaN or infinite losses, validate the prepared dataset:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS_Processed\DataSet\<PREPARED_DATASET_ID>
```

If training cannot find the body model, pass the model path explicitly:

```powershell
--bm-fname <BODY_MODEL_PATH>
```
