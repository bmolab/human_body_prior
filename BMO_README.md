# Modified VPoser Training Workflow

This note covers the current training workflow for the 6D VPoser model and the torque-proxy experiment.

Briefly, the changes we made:
-   The preprocessing was extended for balanced multi-dataset preparation, source metadata, and in-domain/cross-dataset evaluation splits.
-   Updates the original VPoser workflow to use a continuous 6D rotation representation.
-   Adds an exploratory torque-proxy target from AMASS motion sequences.
-   Trains an optional auxiliary prediction head for that proxy.
-   Adds latent-space analysis scripts for checking whether pose and effort-like features are visible in the learned representation.


## 1. Install

Follow the main [`README.md`](README.md).

If you get an error saying it cannot find `human_body_prior`, run: `python setup.py develop`


## 2. Prepare Raw AMASS

Put raw AMASS motion files under:

```text
AMASS\<DATASET_NAME>\...\*_poses.npz
```

The preprocessing uses:

```text
root_orient = poses[:, :3]
pose_body   = poses[:, 3:66]
```

## 3. Preprocess AMASS

Training does not read raw AMASS `.npz` files directly. It reads prepared `.pt` files:

```text
AMASS\DataSet\<PREPARED_DATASET_ID>
  train\pose_body.pt
  train\root_orient.pt
  train\torque_proxy.pt

  vald\pose_body.pt
  vald\root_orient.pt
  vald\torque_proxy.pt

  test\pose_body.pt
  test\root_orient.pt
  test\torque_proxy.pt
```

So you need to start with **setting up a config in `configs\data\`**. A sample config with a detailed explanation can be found here:

- The dataset split is controlled by the `splits` section. 
- If the config lists the same source datasets under `train`, `vald`, and `test`, and enables `split_sequences_by_source`: the preprocessing step then assigns whole source sequences to exactly one in-domain split using the `sequence_split_ratios` ratio, so neighboring frames from the same sequence do not leak across train/validation/test. 
- The **optional** `preprocessing` section keeps torque-proxy training more stable across AMASS sources. It handles outliers (filters non-finite/extreme frames), compresses long-tailed values, and caps frames per sequence/dataset so very large subsets do not dominate.


**Run preprocessing**:

```powershell
python amass_data_prep.py --config configs\data\amass_vposer_in_domain_cross_sample.yaml
```


**Validate the prepared dataset**:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID>
```

Expected result:

- all `finite` is `True`
- `nan_count` is `0`
- `inf_count` is `0`


Prepared datasets include source metadata like:

```text
train\source_dataset.pt
train\source_sequence.pt
source_dataset_names.json
source_sequence_names.json
```

**Note**: When `source_dataset.pt` is present, training logs **validation metrics** both in aggregate and per source dataset, e.g. `vald/CMU/loss_total` and `vald/KIT/loss_total`.



## 4. Training

### Training Inputs

You need:

- a prepared dataset under `AMASS\DataSet\<PREPARED_DATASET_ID>`
- a SMPL/SMPL-X body model `.npz` file

Training experiments live under:

```text
src\human_body_prior\train\<EXPERIMENT_FOLDER>
```

Each experiment must have:

```python
<EXPERIMENT_NAME>.py # the command-line entry point
<EXPERIMENT_NAME>.yaml # stores the default model, data, and training settings.
```


### Torque-Proxy VPoser

Use a dataset prepared with `include_torque_proxy: true`.

Sample command that runs torque-proxy training:

```powershell
python -m human_body_prior.train.V02_08_6D_torque_proxy.V02_08_6D_torque_proxy `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID> `
  --expr-id <EXPR_ID> `
  --bm-fname <BODY_MODEL_PATH>
```

### Training Outputs

Outputs are written under:

```text
support_data\training\training_experiments\<EXPR_ID>
```

Monitor TensorBoard:

```powershell
tensorboard --logdir support_data\training\training_experiments\<EXPR_ID>\tensorboard
```

### Post-Training Analysis

Training performance is mainly evaluated from saved loss history and checkpoints.

Collected during training:

- `val_loss`: the validation `loss_total`, used by early stopping and checkpoint names.
- `train/loss_kl`, `train/matrot`, `train/jtr`, `train/loss_total`
- `vald/loss_kl`, `vald/matrot`, `vald/jtr`, `vald/loss_total`
- `train/torque_proxy` and `vald/torque_proxy` when torque-proxy supervision is enabled.

These values are saved in:

```text
loss_history.csv
loss_history.png
tensorboard\version_*
```

**For best model selection, we currently use the checkpoint with the lowest validation loss**. The selected path is also written to the training log as `best_model_fname`.

### Key Changes From Original VPoser

**The original VPoser** encoder reads axis-angle pose:

```text
21 joints * 3 axis-angle values = 63 values
```

The 6D model converts each joint rotation to a continuous 6D representation before encoding:

```text
21 joints * 6D rotation values = 126 values
```

The decoder outputs 6D rotations, converts them to rotation matrices, and returns axis-angle pose for the body model.

**The torque-proxy experiment uses a separate model**:

```text
src\human_body_prior\models\vposer_torque_model.py
```

This model keeps the same pose encoder/decoder and adds a small prediction head:

```text
poZ_body_mean -> predicted_torque_proxy
```

The decoder still reconstructs pose only. It does not output torque.

### Loss Function Used During Training

Find the loss change in: ` src\human_body_prior\train\vposer_trainer.py`

#### Baseline 6D VPoser

The baseline objective is:

```text
loss_total = loss_kl + matrot + jtr
```

Terms:

- `loss_kl`: keeps the latent distribution close to `N(0, I)`.
- `matrot`: compares reconstructed joint rotations with original rotations.
- `jtr`: compares reconstructed body joints with original body joints after SMPL-X forward kinematics.

Notes:
- `matrot` and `jtr` are included while `current_epoch < keep_extra_loss_terms_until_epoch`.
- `loss_rec_wt` remains in the config, but mesh reconstruction loss is commented out.

Relevant config:

```yaml
train_parms:
  keep_extra_loss_terms_until_epoch: 15
  loss_weights:
    loss_kl_wt: 0.005
    loss_rec_wt: 4
    loss_matrot_wt: 2
    loss_jtr_wt: 2
```

#### **New Torque-Proxy Experiment**

The torque-proxy model keeps the baseline pose autoencoder loss and adds an auxiliary proxy loss:

```text
torque_proxy = loss_torque_proxy_wt * SmoothL1(predicted_torque_proxy, target_torque_proxy)
```

`target_torque_proxy` is computed during preprocessing, from the raw temporal AMASS sequence before random frame sampling. It is an effort-like placeholder based on pose magnitude, pose speed, and pose acceleration.

With torque supervision enabled:

```text
loss_total = loss_kl + matrot + jtr + torque_proxy
```

Relevant config:

```yaml
train_parms:
  torque_proxy_normalize: True
  torque_proxy_normalization: <NORMALIZATION_MODE>
  torque_proxy_loss: <TORQUE_LOSS_TYPE>
  loss_weights:
    loss_torque_proxy_wt: <LOSS_WEIGHT>
```

## Notes for Troubleshooting

If training gives NaN or infinite losses, validate the prepared dataset:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID>
```

If training cannot find the body model, pass the model path explicitly:

```powershell
--bm-fname <BODY_MODEL_PATH>
```
