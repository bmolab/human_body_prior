# Modified VPoser Training Workflow

This note covers the current training workflow for the 6D VPoser model and the torque-proxy experiment.

Run commands from the repository root:


## 1. Install

Follow the main [`README.md`](README.md).

If Python cannot find `human_body_prior`, run: `python setup.py develop`


## 2. Prepare Raw AMASS

Put raw AMASS motion files under:

```text
AMASS\<DATASET_NAME>\...\*_poses.npz
```

Each AMASS `.npz` file must contain `poses`.

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

  vald\pose_body.pt
  vald\root_orient.pt

  test\pose_body.pt
  test\root_orient.pt
```

For torque-proxy training, preprocessing also writes:

```text
train\torque_proxy.pt
vald\torque_proxy.pt
test\torque_proxy.pt
```

Use a config in `configs\data\` with:

```yaml
amass_dir: AMASS
include_torque_proxy: true
out_dir: AMASS/DataSet/<PREPARED_DATASET_ID>
log_name: prepare.log

splits:
  train:
    - <TRAIN_DATASET_NAME>
  vald:
    - <VALIDATION_DATASET_NAME>
  test:
    - <TEST_DATASET_NAME>
```

The split membership is controlled by the `splits` section. If the same AMASS subset is listed under train, validation, and test, that subset is sampled separately for all three splits.

**Run preprocessing**:

```powershell
python amass_data_prep.py --config configs\data\amass_vposer_example.yaml
```

If prepared split folders already exist in `out_dir`, the script asks whether to clean `train`, `vald`, and `test` before rebuilding. After preprocessing, it prints frame counts, percentages, and tensor shapes for each split.

**Validate the prepared dataset**:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID>
```

Expected result:

- all `finite` is `True`
- `nan_count` is `0`
- `inf_count` is `0`

## 4. Training

### Training Inputs

You need:

- a prepared dataset under `AMASS\DataSet\<PREPARED_DATASET_ID>`
- a SMPL/SMPL-X body model `.npz` file

Training experiments live under:

```text
src\human_body_prior\train\<EXPERIMENT_FOLDER>
```

Each experiment has:

```text
<EXPERIMENT_NAME>.py
<EXPERIMENT_NAME>.yaml
```

The `.py` file is the command-line entry point. The `.yaml` file stores the default model, data, and training settings.

### Baseline 6D VPoser

Run baseline 6D training:

```powershell
python -m human_body_prior.train.V02_08_6D.V02_08_6D `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID> `
  --expr-id <EXPR_ID> `
  --num-epochs 30 `
  --batch-size 32 `
  --num-workers 0 `
  --lr 0.0001 `
  --gradient-clip-val 1.0 `
  --bm-fname <BODY_MODEL_PATH>
```

### Torque-Proxy VPoser

Use a dataset prepared with `include_torque_proxy: true`.

Run torque-proxy training:

```powershell
python -m human_body_prior.train.V02_08_6D_torque_proxy.V02_08_6D_torque_proxy `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID> `
  --expr-id <EXPR_ID> `
  --num-epochs 30 `
  --batch-size 32 `
  --num-workers 0 `
  --lr 0.0001 `
  --gradient-clip-val 1.0 `
  --bm-fname <BODY_MODEL_PATH>
```

The torque loss weight can be overridden from the command line:

```powershell
python -m human_body_prior.train.V02_08_6D_torque_proxy.V02_08_6D_torque_proxy `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID> `
  --expr-id <EXPR_ID> `
  --loss-torque-proxy-wt 0.01 `
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

Regenerate the static loss plot:

```powershell
python -m human_body_prior.tools.loss_history `
  support_data\training\training_experiments\<EXPR_ID>\loss_history.csv
```

### Key Changes From Original VPoser

The original VPoser encoder reads axis-angle pose:

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

The loss is implemented in:

```text
src\human_body_prior\train\vposer_trainer.py
```

#### Baseline 6D VPoser

The baseline objective is:

```text
loss_total = loss_kl + matrot + jtr
```

Terms:

- `loss_kl`: keeps the latent distribution close to `N(0, I)`.
- `matrot`: compares reconstructed joint rotations with original rotations.
- `jtr`: compares reconstructed body joints with original body joints after SMPL-X forward kinematics.

Current behavior:

- `loss_kl` is always included.
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

#### Torque-Proxy Experiment

The torque-proxy model keeps the baseline pose autoencoder loss and adds:

```text
torque_proxy = loss_torque_proxy_wt * L1(predicted_torque_proxy, target_torque_proxy)
```

`target_torque_proxy` is computed from the raw temporal AMASS sequence before random frame sampling. It is an effort-like placeholder based on pose magnitude, pose speed, and pose acceleration.

With torque supervision enabled:

```text
loss_total = loss_kl + matrot + jtr + torque_proxy
```

Relevant config:

```yaml
train_parms:
  torque_proxy_normalize: True
  loss_weights:
    loss_torque_proxy_wt: 0.01
```

`torque_proxy_normalize: True` normalizes target torque-proxy values using the training split mean and standard deviation before the L1 loss.

## 5. Troubleshooting

If training gives NaN or infinite losses, validate the prepared dataset:

```powershell
python -m human_body_prior.evaluations.check_vposer_dataset `
  --dataset-dir AMASS\DataSet\<PREPARED_DATASET_ID>
```

If training cannot find the body model, pass the model path explicitly:

```powershell
--bm-fname <BODY_MODEL_PATH>
```
