

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
        help="Prepared VPoser dataset root containing pose_body.pt and torque_proxy.pt files.",
    )
    parser.add_argument(
        "--work-basedir",
        default=osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "../../../../support_data/training/training_experiments",
            )
        ),
        help="Where experiment outputs/checkpoints are written.",
    )
    parser.add_argument("--expr-id", default="V02_08_6D_decoder_torque_proxy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--loss-torque-proxy-wt", type=float, default=None)
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
    print("Decoder torque-proxy training default_ps_fname:", default_ps_fname)

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
    if args.loss_torque_proxy_wt is not None:
        vp_ps.train_parms.loss_weights.loss_torque_proxy_wt = args.loss_torque_proxy_wt

    print("Decoder torque-proxy training dataset_dir:", dataset_dir)
    print("Decoder torque-proxy training work_dir:", osp.join(vp_ps.general.work_basedir, vp_ps.general.expr_id))
    print("Decoder torque-proxy training epochs:", vp_ps.train_parms.num_epochs)
    print("Decoder torque-proxy training batch_size:", vp_ps.train_parms.batch_size)
    print("Decoder torque-proxy training lr:", vp_ps.train_parms.gen_optimizer.args.lr)
    print("Decoder torque-proxy training gradient_clip_val:", vp_ps.train_parms.gradient_clip_val)
    print("Decoder torque-proxy training loss_torque_proxy_wt:", vp_ps.train_parms.loss_weights.loss_torque_proxy_wt)
    print("Decoder torque-proxy training bm_fname:", vp_ps.body_model.bm_fname)

    train_vposer_once(vp_ps.toDict().copy())


if __name__ == "__main__":
    main()

