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
# 2018.01.02

import glob
import os.path as osp
import shutil

import numpy as np
import torch
from configer import Configer
from human_body_prior.tools.omni_tools import logger_sequencer
from human_body_prior.tools.omni_tools import makepath, log2file


def dataset_exists(dataset_dir, split_names=None, data_fields=None):
    '''
    This function checks whether a valid SuperCap dataset directory exists at a location
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    '''
    if dataset_dir is None: return False
    if split_names is None:
        split_names = ['train', 'vald', 'test']
    if data_fields is None:
        data_fields = ['root_orient', 'pose_body']
    import os

    import numpy as np

    done = []
    for split_name in split_names:
        for k in data_fields:  # , 'betas', 'trans', 'joints']:
            outfname = os.path.join(dataset_dir, split_name, f'{k}.pt')
            done.append(os.path.exists(outfname))
    return np.all(done)


def _compute_torque_proxy(poses, mocap_framerate=None, return_components=False):
    pose_body = poses[:, 3:66].astype(np.float32)
    pose_flat = pose_body.reshape(pose_body.shape[0], -1)

    # Use real sequence FPS so temporal proxy values are comparable across AMASS sources.
    fps = 1.0 if mocap_framerate is None else float(np.asarray(mocap_framerate).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 1.0

    pose_speed = np.zeros(pose_body.shape[0], dtype=np.float32)
    pose_speed[1:] = np.linalg.norm(pose_flat[1:] - pose_flat[:-1], axis=-1) * fps

    pose_acceleration = np.zeros_like(pose_speed)
    pose_acceleration[1:] = np.abs(pose_speed[1:] - pose_speed[:-1]) * fps

    pose_magnitude = np.linalg.norm(pose_body.reshape(pose_body.shape[0], 21, 3), axis=-1).mean(axis=-1)
    torque_proxy = (pose_magnitude * (pose_speed + pose_acceleration)).astype(np.float32)
    if return_components:
        return torque_proxy, pose_speed, pose_acceleration
    return torque_proxy


def prepare_vposer_datasets(
    vposer_dataset_dir,
    amass_splits,
    amass_dir,
    logger=None,
    include_torque_proxy=False,
    keep_rate=0.3,
    max_frames_per_sequence=None,
    max_frames_per_dataset=None,
    torque_proxy_use_fps=True,
    torque_proxy_log1p=True,
    torque_proxy_filter_percentile=99.5,
):
    data_fields = ['root_orient', 'pose_body']
    if include_torque_proxy:
        data_fields.append('torque_proxy')

    if dataset_exists(vposer_dataset_dir, data_fields=data_fields):
        if logger is not None: logger(f'VPoser dataset already exists at {vposer_dataset_dir}')
        return

    ds_logger = log2file(makepath(vposer_dataset_dir, 'dataset.log', isfile=True), write2file_only=True)
    logger = ds_logger if logger is None else logger_sequencer([ds_logger, logger])

    logger(f'Creating pytorch dataset at {vposer_dataset_dir}')
    logger(f'Using AMASS body parameters from {amass_dir}')

    shutil.copy2(__file__, vposer_dataset_dir)

    # class AMASS_ROW(pytables.IsDescription):
    #
    #     # gender = pytables.Int16Col(1)  # 1-character String
    #     root_orient = pytables.Float32Col(3)  # float  (single-precision)
    #     pose_body = pytables.Float32Col(21 * 3)  # float  (single-precision)
    #     # pose_hand = pytables.Float32Col(2 * 15 * 3)  # float  (single-precision)
    #
    #     # betas = pytables.Float32Col(16)  # float  (single-precision)
    #     # trans = pytables.Float32Col(3)  # float  (single-precision)

    def _finite_pose_mask(poses):
        # Drop broken frames before sampling them into the training tensors.
        return np.isfinite(poses[:, :66]).all(axis=1)

    def _choose_sample_ids(candidate_ids, remaining_dataset_budget=None):
        if len(candidate_ids) < 1:
            return candidate_ids

        sample_count = int(keep_rate * len(candidate_ids))
        sample_count = max(1, sample_count)
        # Caps keep very large sequences or datasets from dominating the split.
        if max_frames_per_sequence is not None:
            sample_count = min(sample_count, int(max_frames_per_sequence))
        if remaining_dataset_budget is not None:
            sample_count = min(sample_count, int(remaining_dataset_budget))
        sample_count = min(sample_count, len(candidate_ids))
        if sample_count < 1:
            return np.asarray([], dtype=np.int64)
        return np.random.choice(candidate_ids, sample_count, replace=False)

    def fetch_from_amass(ds_names):
        for ds_name in ds_names:
            mosh_stageII_fnames = glob.glob(osp.join(amass_dir, ds_name, '*/*_poses.npz'))
            logger('Found {} sequences from {}.'.format(len(mosh_stageII_fnames), ds_name))

            ds_frame_count = 0
            for npz_fname in np.random.permutation(mosh_stageII_fnames):
                if max_frames_per_dataset is not None and ds_frame_count >= int(max_frames_per_dataset):
                    logger('Reached max_frames_per_dataset={} for {}.'.format(max_frames_per_dataset, ds_name))
                    break

                cdata = np.load(npz_fname)
                N = len(cdata['poses'])

                # skip first and last frames to avoid initial standard poses, e.g. T pose
                candidate_ids = np.asarray(list(range(int(0.1 * N), int(0.9 * N), 1)), dtype=np.int64)
                if len(candidate_ids) < 1:
                    continue
                poses = cdata['poses'].astype(np.float32)
                valid_mask = _finite_pose_mask(poses)
                torque_proxy = None

                if include_torque_proxy:
                    mocap_framerate = None
                    if torque_proxy_use_fps and 'mocap_framerate' in cdata.files:
                        mocap_framerate = cdata['mocap_framerate']
                    torque_proxy, pose_speed, pose_acceleration = _compute_torque_proxy(
                        poses,
                        mocap_framerate=mocap_framerate,
                        return_components=True,
                    )
                    valid_mask &= np.isfinite(torque_proxy)
                    valid_mask &= np.isfinite(pose_speed)
                    valid_mask &= np.isfinite(pose_acceleration)

                    # Remove extreme proxy spikes caused by noisy temporal jumps.
                    finite_proxy = torque_proxy[np.isfinite(torque_proxy)]
                    if torque_proxy_filter_percentile is not None and len(finite_proxy) > 0:
                        proxy_limit = np.percentile(finite_proxy, float(torque_proxy_filter_percentile))
                        valid_mask &= torque_proxy <= proxy_limit

                candidate_ids = candidate_ids[valid_mask[candidate_ids]]
                remaining_dataset_budget = None
                if max_frames_per_dataset is not None:
                    remaining_dataset_budget = int(max_frames_per_dataset) - ds_frame_count
                cdata_ids = _choose_sample_ids(candidate_ids, remaining_dataset_budget)
                if len(cdata_ids) < 1: continue
                fullpose = poses[cdata_ids]
                data = {'pose_body': fullpose[:, 3:66], 'root_orient': fullpose[:, :3]}
                if include_torque_proxy:
                    selected_torque_proxy = torque_proxy[cdata_ids]
                    if torque_proxy_log1p:
                        # Compress the long-tailed proxy scale before training normalization.
                        selected_torque_proxy = np.log1p(np.maximum(selected_torque_proxy, 0.0))
                    data['torque_proxy'] = selected_torque_proxy.astype(np.float32)
                ds_frame_count += len(cdata_ids)
                yield data

    for split_name, ds_names in amass_splits.items():
        if dataset_exists(vposer_dataset_dir, split_names=[split_name], data_fields=data_fields): continue
        logger(f'Preparing VPoser data for split {split_name}')

        data_fields = {}
        for data in fetch_from_amass(ds_names):
            for k in data.keys():
                if k not in data_fields: data_fields[k] = []
                data_fields[k].append(data[k])

        for k, v in data_fields.items():
            outpath = makepath(vposer_dataset_dir, split_name, '{}.pt'.format(k), isfile=True)
            v = np.concatenate(v)
            torch.save(torch.tensor(v), outpath)

        logger(
            f'{len(v)} datapoints dumped for split {split_name}. ds_meta_pklpath: {osp.join(vposer_dataset_dir, split_name)}')

    Configer(**{
        'amass_splits': amass_splits.toDict(),
        'amass_dir': amass_dir,
    }).dump_settings(makepath(vposer_dataset_dir, 'settings.ini', isfile=True))

    logger(f'Dumped final pytorch dataset at {vposer_dataset_dir}')
