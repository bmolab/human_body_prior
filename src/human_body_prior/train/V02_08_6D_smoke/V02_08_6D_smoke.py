r"""Small 6DOF VPoser smoke-test training entry point.

This is intended to validate wiring after code changes, not to train a final
model. It uses a tiny epoch count, small batch size, no rendering, and a prepared
dataset path supplied from the command line.

Example:
    python -m human_body_prior.train.V02_08_6D_smoke.V02_08_6D_smoke \
        --dataset-dir D:\Git\human_body_prior\AMASS\DataSet\SFU_test
"""

import argparse
import glob
import os.path as osp

from human_body_prior.tools.configurations import load_config
from human_body_prior.train.vposer_trainer import train_vposer_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared VPoser dataset root containing train/vald/test pose_body.pt files.",
    )
    parser.add_argument(
        "--work-basedir",
        default=osp.abspath(osp.join(osp.dirname(__file__), "../../../../support_data/training/training_experiments")),
        help="Where smoke-test experiment outputs/checkpoints are written.",
    )
    parser.add_argument("--expr-id", default="V02_08_6D_smoke")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--bm-fname",
        default=osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "../../../../support_data/dowloads/models/smplx/neutral/SMPLX_NEUTRAL.npz",
            )
        ),
        help="Body model .npz file used for joint/mesh losses.",
    )
    args = parser.parse_args()

    default_ps_fname = glob.glob(osp.join(osp.dirname(__file__), "*.yaml"))[0]
    vp_ps = load_config(default_ps_fname)

    dataset_dir = osp.abspath(args.dataset_dir)
    vp_ps.general.dataset_basedir = osp.dirname(dataset_dir)
    vp_ps.general.dataset_id = osp.basename(dataset_dir)
    vp_ps.general.work_basedir = osp.abspath(args.work_basedir)
    vp_ps.general.expr_id = args.expr_id
    vp_ps.body_model.bm_fname = osp.abspath(args.bm_fname)

    vp_ps.train_parms.batch_size = args.batch_size
    vp_ps.train_parms.num_epochs = args.num_epochs
    vp_ps.train_parms.gen_optimizer.args.lr = args.lr
    vp_ps.train_parms.gradient_clip_val = args.gradient_clip_val
    vp_ps.data_parms.num_workers = args.num_workers
    vp_ps.logging.render_during_training = False

    print("Smoke-test dataset_dir:", dataset_dir)
    print("Smoke-test work_dir:", osp.join(vp_ps.general.work_basedir, vp_ps.general.expr_id))
    print("Smoke-test epochs:", vp_ps.train_parms.num_epochs)
    print("Smoke-test batch_size:", vp_ps.train_parms.batch_size)
    print("Smoke-test lr:", vp_ps.train_parms.gen_optimizer.args.lr)
    print("Smoke-test gradient_clip_val:", vp_ps.train_parms.gradient_clip_val)
    print("Smoke-test bm_fname:", vp_ps.body_model.bm_fname)

    train_vposer_once(vp_ps.toDict().copy())


if __name__ == "__main__":
    main()
