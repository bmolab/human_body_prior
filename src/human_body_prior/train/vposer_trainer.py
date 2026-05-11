# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

# from pytorch_lightning import Trainer

import glob
import inspect
import json
import os
import os.path as osp
import subprocess
from datetime import datetime as dt
from typing import Any

from numba.cuda import gpus
from pytorch_lightning.strategies import DDPStrategy

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.data.prepare_data import dataset_exists
from human_body_prior.data.prepare_data import prepare_vposer_datasets
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.models.vposer_torque_model import VPoserTorqueProxy
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from human_body_prior.tools.configurations import load_config, dump_config
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import log2file
from human_body_prior.tools.omni_tools import make_deterministic
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.tools.loss_history import LossHistoryRecorder, plot_loss_history
from human_body_prior.visualizations.training_visualization import vposer_trainer_renderer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.utils.data import DataLoader

resume_from_checkpoint = None

class VPoserTrainer(LightningModule):
    """

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """

    def __init__(self, _config):
        super(VPoserTrainer, self).__init__()

        _support_data_dir = get_support_data_dir()

        vp_ps = load_config(**_config)
        self.fast_dev_run = True

        make_deterministic(vp_ps.general.rnd_seed)

#--- DR added
        self.training_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []   # save targets in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []        # save targets in each batch to compute metric overall epoch
#--- end

        self.expr_id = vp_ps.general.expr_id
        self.dataset_id = vp_ps.general.dataset_id

        self.work_dir = vp_ps.logging.work_dir = makepath(vp_ps.general.work_basedir, self.expr_id)
        self.dataset_dir = vp_ps.logging.dataset_dir = osp.join(vp_ps.general.dataset_basedir, vp_ps.general.dataset_id)
        self.source_dataset_names = self._load_source_dataset_names()

        self._log_prefix = '[{}]'.format(self.expr_id)
        self.text_logger = log2file(prefix=self._log_prefix)
        self.loss_history = LossHistoryRecorder(self.work_dir, logger=self.text_logger)

        self.seq_len = vp_ps.data_parms.num_timeseq_frames
        self.use_torque_proxy = self._uses_torque_proxy(vp_ps)
        self.torque_proxy_normalize = bool(vp_ps.train_parms.toDict().get('torque_proxy_normalize', True))
        # Torque labels can be long-tailed, so these options make scaling/loss robust.
        self.torque_proxy_normalization = vp_ps.train_parms.toDict().get('torque_proxy_normalization', 'mean_std')
        self.torque_proxy_loss = vp_ps.train_parms.toDict().get('torque_proxy_loss', 'smooth_l1')
        self.torque_proxy_smooth_l1_beta = float(vp_ps.train_parms.toDict().get('torque_proxy_smooth_l1_beta', 1.0))
        self._torque_proxy_stats_ready = False
        self.register_buffer('torque_proxy_mean', torch.zeros(1))
        self.register_buffer('torque_proxy_std', torch.ones(1))

        vp_model_class = VPoserTorqueProxy if self.use_torque_proxy else VPoser
        self.vp_model = vp_model_class(vp_ps)

        with torch.no_grad():
            self.bm_train = BodyModel(vp_ps.body_model.bm_fname)

        if vp_ps.logging.render_during_training:
            self.renderer = vposer_trainer_renderer(self.bm_train, vp_ps.logging.num_bodies_to_display)
        else:
            self.renderer = None

        self.example_input_array = {'pose_body':torch.ones(vp_ps.train_parms.batch_size, 63),}
        self.vp_ps = vp_ps

    def forward(self, pose_body):

        return self.vp_model(pose_body)

    @staticmethod
    def _uses_torque_proxy(vp_ps):
        loss_weights = vp_ps.train_parms.loss_weights.toDict()
        return float(loss_weights.get('loss_torque_proxy_wt', 0.0)) > 0.0

    def _load_source_dataset_names(self):
        source_names_fname = osp.join(self.dataset_dir, 'source_dataset_names.json')
        if not osp.exists(source_names_fname):
            return []
        with open(source_names_fname, 'r') as source_names_file:
            return json.load(source_names_file)

    def _ensure_torque_proxy_stats(self):
        if not self.use_torque_proxy or not self.torque_proxy_normalize or self._torque_proxy_stats_ready:
            return

        torque_fname = osp.join(self.dataset_dir, 'train', 'torque_proxy.pt')
        torque_proxy = torch.load(torque_fname).type(torch.float32).view(-1)
        finite_mask = torch.isfinite(torque_proxy)
        torque_proxy = torque_proxy[finite_mask]
        if torque_proxy.numel() == 0:
            raise ValueError('No finite torque_proxy values found in {}'.format(torque_fname))

        # Robust mode uses median/IQR so a few high-effort frames do not set the scale.
        if self.torque_proxy_normalization in ('robust', 'median_iqr'):
            q25 = torch.quantile(torque_proxy, 0.25)
            q50 = torch.quantile(torque_proxy, 0.50)
            q75 = torch.quantile(torque_proxy, 0.75)
            torque_center = q50
            torque_scale = q75 - q25
        else:
            torque_center = torque_proxy.mean()
            torque_scale = torque_proxy.std(unbiased=False)

        if torque_scale.item() < 1e-8:
            torque_scale = torch.ones_like(torque_scale)

        self.torque_proxy_mean.copy_(torque_center.to(self.torque_proxy_mean.device).view(1))
        self.torque_proxy_std.copy_(torque_scale.to(self.torque_proxy_std.device).view(1))
        self._torque_proxy_stats_ready = True

    def _get_data(self, split_name):

        assert split_name in ('train', 'vald', 'test')

        split_name = split_name.replace('vald', 'vald')

        required_fields = ['root_orient', 'pose_body']
        data_fields = ['pose_body']
        if self.use_torque_proxy:
            required_fields.append('torque_proxy')
            data_fields.append('torque_proxy')
        if osp.exists(osp.join(self.dataset_dir, split_name, 'source_dataset.pt')):
            data_fields.append('source_dataset')
            if not self.source_dataset_names:
                self.source_dataset_names = self._load_source_dataset_names()

        assert dataset_exists(self.dataset_dir, data_fields=required_fields), FileNotFoundError('Dataset does not exist dataset_dir = {}'.format(self.dataset_dir))
        dataset = VPoserDS(osp.join(self.dataset_dir, split_name), data_fields=data_fields)
        self._ensure_torque_proxy_stats()

        assert len(dataset) != 0, ValueError('Dataset has nothing in it!')

        return DataLoader(dataset,
                          batch_size=self.vp_ps.train_parms.batch_size,
                          shuffle=True if split_name == 'train' else False,
                          num_workers=self.vp_ps.data_parms.num_workers,
                          pin_memory=True)

    @rank_zero_only
    def on_train_start(self):
        if self.global_rank != 0: return
        self.train_starttime = dt.now().replace(microsecond=0)

        ######## make a backup of vposer
        git_repo_dir = os.path.abspath(__file__)
        while True:
            if os.path.isdir(os.path.join(git_repo_dir, '.git')):
                break
            parent = os.path.dirname(git_repo_dir)
            if parent == git_repo_dir:
                git_repo_dir = os.getcwd()
                break
            git_repo_dir = parent
        starttime = dt.strftime(self.train_starttime, '%Y_%m_%d_%H_%M_%S')
        archive_path = makepath(self.work_dir, 'code', 'vposer_{}.tar.gz'.format(starttime), isfile=True)
        try:
            subprocess.run(
                ['git', '-C', git_repo_dir, 'archive', '--format=tar.gz', '-o', archive_path, 'HEAD'],
                check=True,
            )
        except Exception as exc:
            self.text_logger('Skipping git archive backup: {}'.format(exc))
        ########
        if osp.exists(archive_path):
            self.text_logger('Created a git archive backup at {}'.format(archive_path))
        dump_config(self.vp_ps, osp.join(self.work_dir, '{}.yaml'.format(self.expr_id)))

    def train_dataloader(self):
        return self._get_data('train')

    def val_dataloader(self):
        return self._get_data('vald')

    def configure_optimizers(self):
        params_count = lambda params: sum(p.numel() for p in params if p.requires_grad)

        gen_params = [a[1] for a in self.vp_model.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.vp_ps.train_parms.gen_optimizer.type)
        gen_optimizer = gen_optimizer_class(gen_params, **self.vp_ps.train_parms.gen_optimizer.args)

        self.text_logger('Total Trainable Parameters Count in vp_model is %2.2f M.' % (params_count(gen_params) * 1e-6))

        lr_sched_class = getattr(lr_sched_module, self.vp_ps.train_parms.lr_scheduler.type)

        lr_scheduler_args = self.vp_ps.train_parms.lr_scheduler.args.toDict()
        scheduler_sig = inspect.signature(lr_sched_class.__init__)
        lr_scheduler_args = {k: v for k, v in lr_scheduler_args.items() if k in scheduler_sig.parameters}
        gen_lr_scheduler = lr_sched_class(gen_optimizer, **lr_scheduler_args)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
        ]
        return [gen_optimizer], schedulers

    def _compute_loss(self, dorig, drec, return_sample_loss=False):
        geodesic_loss_per_joint = geodesic_loss_R(reduction='none')

        bs, latentD = drec['poZ_body_mean'].shape
        device = drec['poZ_body_mean'].device

        loss_kl_wt = self.vp_ps.train_parms.loss_weights.loss_kl_wt
        loss_rec_wt = self.vp_ps.train_parms.loss_weights.loss_rec_wt
        loss_matrot_wt = self.vp_ps.train_parms.loss_weights.loss_matrot_wt
        loss_jtr_wt = self.vp_ps.train_parms.loss_weights.loss_jtr_wt
        loss_torque_proxy_wt = self.vp_ps.train_parms.loss_weights.toDict().get('loss_torque_proxy_wt', 0.0)

        # q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        q_z = drec['q_z']
        # dorig['fullpose'] = torch.cat([dorig['root_orient'], dorig['pose_body']], dim=-1)

        # Reconstruction loss - L1 on the output mesh (input mesh vs output mesh)
        with torch.no_grad():
            bm_orig = self.bm_train(pose_body=dorig['pose_body'])

        bm_rec = self.bm_train(pose_body=drec['pose_body'].contiguous().view(bs, -1))

        # remove calculation since we are not doing mesh loss, only JTr loss
        # v2v = l1_loss(bm_rec.v, bm_orig.v)


        # KL loss (this enforces unit sphere (?) distribution for VAE... deviation of q_z (encoded latent) from p_z (preferred distribution)
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones((bs, latentD), device=device, requires_grad=False))
        sample_loss_dict = {
            'loss_kl': loss_kl_wt * torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])
        }

        weighted_loss_dict = {
            'loss_kl': torch.mean(sample_loss_dict['loss_kl']) #,
            # 'loss_mesh_rec': loss_rec_wt * v2v
        }

        if (self.current_epoch < self.vp_ps.train_parms.keep_extra_loss_terms_until_epoch):
            # breakpoint()
            matrot_per_joint = geodesic_loss_per_joint(
                drec['pose_body_matrot'].view(-1, 3, 3),
                aa2matrot(dorig['pose_body'].view(-1, 3)),
            ).view(bs, -1)
            sample_loss_dict['matrot'] = loss_matrot_wt * matrot_per_joint.mean(dim=1)
            sample_loss_dict['jtr'] = loss_jtr_wt * torch.mean(
                torch.abs(bm_rec.Jtr - bm_orig.Jtr).view(bs, -1),
                dim=1,
            )
            weighted_loss_dict['matrot'] = sample_loss_dict['matrot'].mean()
            weighted_loss_dict['jtr'] = sample_loss_dict['jtr'].mean()

        if self.use_torque_proxy:
            target_torque_proxy = dorig['torque_proxy'].view(-1).to(device)
            if self.torque_proxy_normalize:
                target_torque_proxy = (target_torque_proxy - self.torque_proxy_mean) / (self.torque_proxy_std + 1e-8)
            # SmoothL1 is less sensitive to noisy proxy targets than plain L1.
            if self.torque_proxy_loss.lower() in ('smooth_l1', 'huber'):
                torque_loss = torch.nn.functional.smooth_l1_loss(
                    drec['pred_torque_proxy'].view(-1),
                    target_torque_proxy,
                    beta=self.torque_proxy_smooth_l1_beta,
                    reduction='none',
                )
            else:
                torque_loss = torch.abs(drec['pred_torque_proxy'].view(-1) - target_torque_proxy)
            sample_loss_dict['torque_proxy'] = loss_torque_proxy_wt * torque_loss
            weighted_loss_dict['torque_proxy'] = sample_loss_dict['torque_proxy'].mean()

        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()
        sample_loss_dict['loss_total'] = torch.stack(list(sample_loss_dict.values())).sum(dim=0)

        # with torch.no_grad():
        #     # unweighted_loss_dict = {'v2v': torch.sqrt(torch.pow(bm_rec.v-bm_orig.v, 2).sum(-1)).mean()}
        #     unweighted_loss_dict = {}
        #     unweighted_loss_dict['loss_total'] = torch.cat(
        #         list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)

        if return_sample_loss:
            return {'weighted_loss': weighted_loss_dict, 'sample_loss': sample_loss_dict}
        return {'weighted_loss': weighted_loss_dict}
        # return {'weighted_loss': weighted_loss_dict, 'unweighted_loss': unweighted_loss_dict}

    def training_step(self, batch, batch_idx):
        # process the data
        drec = self(batch['pose_body'].view(-1, 63))        # batch is incoming data (axis angle), drec is reconstructed data
        # out_pose = drec['pose_body'][0]
        # in_pose = batch['pose_body'][0].view(21, -1)
        loss = self._compute_loss(batch, drec)
        self.loss_history.update('train', loss['weighted_loss'])
        self._log_loss_components('train', loss['weighted_loss'])

        train_loss = loss['weighted_loss']['loss_total']

        tensorboard_logs = {'train_loss': train_loss}
        progress_bar = {k: c2c(v) for k, v in loss['weighted_loss'].items()}
        return {'loss': train_loss, 'progress_bar':progress_bar,  'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        drec = self(batch['pose_body'].view(-1, 63))

        loss = self._compute_loss(batch, drec, return_sample_loss='source_dataset' in batch)
        if not self._is_sanity_checking():
            self.loss_history.update('vald', loss['weighted_loss'])
        self._log_loss_components('vald', loss['weighted_loss'])
        if 'source_dataset' in batch:
            self._log_source_loss_components('vald', loss['sample_loss'], batch['source_dataset'])
        val_loss = loss['weighted_loss']['loss_total']
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.renderer is not None and self.global_rank == 0 and batch_idx % 500==0 and np.random.rand()>0.5:
            out_fname = makepath(self.work_dir, 'renders/vald_rec_E{:03d}_It{:04d}_val_loss_{:.2f}.png'.format(self.current_epoch, batch_idx, val_loss.item()), isfile=True)
            self.renderer([batch, drec], out_fname = out_fname)
            dgen = self.vp_model.sample_poses(self.vp_ps.logging.num_bodies_to_display)
            out_fname = makepath(self.work_dir, 'renders/vald_gen_E{:03d}_I{:04d}.png'.format(self.current_epoch, batch_idx), isfile=True)
            self.renderer([dgen], out_fname = out_fname)


        progress_bar = {'v2v': val_loss}
        return {'val_loss': c2c(val_loss), 'progress_bar': progress_bar, 'log': progress_bar}

    def _log_loss_components(self, split_name, weighted_loss):
        for loss_name, loss_value in weighted_loss.items():
            self.log(
                '{}/{}'.format(split_name, loss_name),
                loss_value.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    def _log_source_loss_components(self, split_name, sample_loss, source_dataset):
        source_dataset = source_dataset.view(-1).to(sample_loss['loss_total'].device).long()
        for source_id in torch.unique(source_dataset):
            source_id_int = int(source_id.item())
            source_name = self._source_dataset_name(source_id_int)
            source_mask = source_dataset == source_id
            source_count = int(source_mask.sum().item())
            if source_count < 1:
                continue
            for loss_name, loss_value in sample_loss.items():
                self.log(
                    '{}/{}/{}'.format(split_name, source_name, loss_name),
                    loss_value[source_mask].mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=source_count,
                )

    def _source_dataset_name(self, source_id):
        if 0 <= source_id < len(self.source_dataset_names):
            return self.source_dataset_names[source_id]
        return 'source_{}'.format(source_id)

    def on_train_epoch_start(self):
        self.loss_history.reset('train')

    def on_validation_epoch_start(self):
        if not self._is_sanity_checking():
            self.loss_history.reset('vald')

    @rank_zero_only
    def on_train_epoch_end(self):
        self.loss_history.write_epoch('train', self.current_epoch)

    @rank_zero_only
    def on_validation_epoch_end(self):
        if not self._is_sanity_checking():
            self.loss_history.write_epoch('vald', self.current_epoch)

    def _is_sanity_checking(self):
        return getattr(self.trainer, 'sanity_checking', False)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        loss = outputs['loss'].item()

        # if self.global_rank == 0:
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.text_logger(str(loss))
            # self.text_logger('lr is {}'.format([pg['lr'] for opt in self.trainer.optimizers for pg in opt.param_groups]))

        # metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}

        # return {'val_loss': metrics['val_loss'], 'log': metrics}

    # def on_validation_epoch_end(self):
    # def validation_epoch_end(self, outputs):
    #     metrics = {'val_loss': np.nanmean(np.concatenate([v['val_loss'] for v in outputs])) }
    #
    #     if self.global_rank == 0:
    #
    #         self.text_logger('Epoch {}: {}'.format(self.current_epoch, ', '.join('{}:{:.2f}'.format(k, v) for k, v in metrics.items())))
    #         self.text_logger('lr is {}'.format([pg['lr'] for opt in self.trainer.optimizers for pg in opt.param_groups]))
    #
    #     metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}
    #
    #     return {'val_loss': metrics['val_loss'], 'log': metrics}


    @rank_zero_only
    def on_train_end(self):

        self.train_endtime = dt.now().replace(microsecond=0)
        endtime = dt.strftime(self.train_endtime, '%Y_%m_%d_%H_%M_%S')
        elapsedtime = self.train_endtime - self.train_starttime
        self.vp_ps.logging.best_model_fname = self.trainer.checkpoint_callback.best_model_path

        self.text_logger('Epoch {} - Finished training at {} after {}'.format(self.current_epoch, endtime, elapsedtime))
        self.text_logger('best_model_fname: {}'.format(self.vp_ps.logging.best_model_fname))
        try:
            plot_path = plot_loss_history(self.loss_history.csv_path)
            self.text_logger('loss_history_plot: {}'.format(plot_path))
        except Exception as exc:
            self.text_logger('Could not plot loss history: {}'.format(exc))

        dump_config(self.vp_ps, osp.join(self.work_dir, '{}_{}.yaml'.format(self.expr_id, self.dataset_id)))
        # Newer PyTorch Lightning exposes hparams as a read-only property.
        # The final experiment config is already persisted above.

    @rank_zero_only
    def prepare_data(self):
        '''' Similar to standard AMASS dataset preparation pipeline:
        Donwload npz file, corresponding to body data from https://amass.is.tue.mpg.de/ and place them under amass_dir
        '''
        self.text_logger = log2file(makepath(self.work_dir, '{}.log'.format(self.expr_id), isfile=True), prefix=self._log_prefix)

        prepare_vposer_datasets(
            self.dataset_dir,
            self.vp_ps.data_parms.amass_splits,
            self.vp_ps.data_parms.amass_dir,
            logger=self.text_logger,
            include_torque_proxy=self.use_torque_proxy,
        )


def create_expr_message(ps):
    expr_msg = '[{}] batch_size = {}.'.format(ps.general.expr_id, ps.train_parms.batch_size)

    return expr_msg


def train_vposer_once(_config):

    resume_training_if_possible = True

    model = VPoserTrainer(_config)
    model.vp_ps.logging.expr_msg = create_expr_message(model.vp_ps)
    # model.text_logger(model.vp_ps.logging.expr_msg.replace(". ", '.\n'))
    dump_config(model.vp_ps, osp.join(model.work_dir, '{}.yaml'.format(model.expr_id)))

    logger = TensorBoardLogger(model.work_dir, name='tensorboard')
    lr_monitor = LearningRateMonitor()

    snapshots_dir = osp.join(model.work_dir, 'snapshots')
    checkpoint_callback = ModelCheckpoint(
        dirpath=makepath(snapshots_dir, isfile=True),
        filename="%s_{epoch:02d}_{val_loss:.2f}" % model.expr_id,
        save_top_k=-1,
        verbose=True,
        every_n_epochs=1,
    )
    early_stop_callback = EarlyStopping(**model.vp_ps.train_parms.early_stopping)

    resume_from_checkpoint = None
    if resume_training_if_possible:
        available_ckpts = sorted(glob.glob(osp.join(snapshots_dir, '*.ckpt')), key=os.path.getmtime)
        if len(available_ckpts)>0:
            resume_from_checkpoint = available_ckpts[-1]
            model.text_logger('Resuming the training from {}'.format(resume_from_checkpoint))

    gradient_clip_val = getattr(model.vp_ps.train_parms, 'gradient_clip_val', 0.0)

    trainer = pl.Trainer( # gpus=1,
                         # weights_summary='top',
                         # distributed_backend = 'ddp',
                         # replace_sampler_ddp=False,
                         # accumulate_grad_batches=4,
                         # profiler=False,
                         # overfit_batches=0.05,
                         # fast_dev_run = True,
                         # limit_train_batches=0.02,
                         # limit_val_batches=0.02,
                         # num_sanity_val_steps=2,
                         # strategy=DDPStrategy(),
                         accelerator='gpu',
                         callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
                         gradient_clip_val=gradient_clip_val,

                         max_epochs=model.vp_ps.train_parms.num_epochs,
                         logger=logger,
                         # resume_from_checkpoint=resume_from_checkpoint
                         )

    trainer.fit(model)
    # trainer.fit(model, ckpt_path='/Users/drokeby/Dev/human_body_prior_/support_data/training/training_experiments/V02_07/snapshots/V02_07_epoch=00_val_loss=0.00.ckpt')
