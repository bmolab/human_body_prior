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
import json
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


def find_amass_motion_files(amass_dir, ds_name):
    """Find AMASS motion npz files across common release naming schemes."""

    patterns = ['*/*_poses.npz', '*/*_stageii.npz']
    motion_files = []
    for pattern in patterns:
        motion_files.extend(glob.glob(osp.join(amass_dir, ds_name, pattern)))
    return sorted(set(motion_files))


def estimate_amass_sequence_sample_count(
    npz_fname,
    keep_rate=0.3,
    max_frames_per_sequence=None,
    include_torque_proxy=False,
    torque_proxy_use_fps=True,
    torque_proxy_filter_percentile=99.5,
):
    with np.load(npz_fname) as cdata:
        poses = cdata['poses'].astype(np.float32)
        n_frames = len(poses)
        candidate_ids = np.asarray(list(range(int(0.1 * n_frames), int(0.9 * n_frames), 1)), dtype=np.int64)
        if len(candidate_ids) < 1:
            return 0

        valid_mask = np.isfinite(poses[:, :66]).all(axis=1)
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

            finite_proxy = torque_proxy[np.isfinite(torque_proxy)]
            if torque_proxy_filter_percentile is not None and len(finite_proxy) > 0:
                proxy_limit = np.percentile(finite_proxy, float(torque_proxy_filter_percentile))
                valid_mask &= torque_proxy <= proxy_limit

        valid_count = int(valid_mask[candidate_ids].sum())
        if valid_count < 1:
            return 0

        sample_count = max(1, int(keep_rate * valid_count))
        if max_frames_per_sequence is not None:
            sample_count = min(sample_count, int(max_frames_per_sequence))
        return max(0, min(sample_count, valid_count))


def build_frame_balanced_sequence_assignments(
    amass_dir,
    amass_splits,
    split_sequences_by_source=False,
    sequence_split_ratios=None,
    sequence_split_seed=100,
    keep_rate=0.3,
    max_frames_per_sequence=None,
    include_torque_proxy=False,
    torque_proxy_use_fps=True,
    torque_proxy_filter_percentile=99.5,
):
    source_dataset_names = []
    for split_ds_names in amass_splits.values():
        for ds_name in split_ds_names:
            if ds_name not in source_dataset_names:
                source_dataset_names.append(ds_name)

    assignments = {}
    split_ratios = sequence_split_ratios or {'train': 0.8, 'vald': 0.1, 'test': 0.1}
    rng = np.random.RandomState(int(sequence_split_seed))
    for ds_name in source_dataset_names:
        requested_splits = [
            split_name for split_name, ds_names in amass_splits.items()
            if ds_name in ds_names
        ]
        npz_fnames = find_amass_motion_files(amass_dir, ds_name)
        if split_sequences_by_source and len(requested_splits) > 1:
            sequence_infos = []
            for npz_fname in npz_fnames:
                sequence_infos.append((
                    npz_fname,
                    estimate_amass_sequence_sample_count(
                        npz_fname,
                        keep_rate=keep_rate,
                        max_frames_per_sequence=max_frames_per_sequence,
                        include_torque_proxy=include_torque_proxy,
                        torque_proxy_use_fps=torque_proxy_use_fps,
                        torque_proxy_filter_percentile=torque_proxy_filter_percentile,
                    ),
                    rng.rand(),
                ))

            total_frames = float(sum(frame_count for _, frame_count, _ in sequence_infos))
            split_weights = np.asarray(
                [float(split_ratios.get(split_name, 1.0)) for split_name in requested_splits],
                dtype=np.float64,
            )
            if split_weights.sum() <= 0:
                split_weights = np.ones(len(requested_splits), dtype=np.float64)
            split_weights = split_weights / split_weights.sum()
            targets = {
                split_name: total_frames * split_weight
                for split_name, split_weight in zip(requested_splits, split_weights)
            }
            current = {split_name: 0.0 for split_name in requested_splits}
            for split_name in requested_splits:
                assignments[(split_name, ds_name)] = []

            for npz_fname, frame_count, _ in sorted(sequence_infos, key=lambda item: (-item[1], item[2])):
                target_split = max(
                    requested_splits,
                    key=lambda split_name: targets[split_name] - current[split_name],
                )
                assignments[(target_split, ds_name)].append(npz_fname)
                current[target_split] += float(frame_count)
        else:
            for split_name in requested_splits:
                assignments[(split_name, ds_name)] = npz_fnames
    return assignments


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
    save_source_metadata=False,
    split_sequences_by_source=False,
    sequence_split_ratios=None,
    sequence_split_seed=100,
):
    required_data_fields = ['root_orient', 'pose_body']
    if include_torque_proxy:
        required_data_fields.append('torque_proxy')
    if save_source_metadata:
        required_data_fields.extend(['source_dataset', 'source_sequence'])

    if dataset_exists(vposer_dataset_dir, split_names=list(amass_splits.keys()), data_fields=required_data_fields):
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

    def _source_dataset_names():
        ds_names = []
        for split_ds_names in amass_splits.values():
            for ds_name in split_ds_names:
                if ds_name not in ds_names:
                    ds_names.append(ds_name)
        return ds_names

    source_dataset_names = _source_dataset_names()
    source_dataset_to_id = {ds_name: ds_id for ds_id, ds_name in enumerate(source_dataset_names)}

    sequence_assignments = build_frame_balanced_sequence_assignments(
        amass_dir,
        amass_splits,
        split_sequences_by_source=split_sequences_by_source,
        sequence_split_ratios=sequence_split_ratios,
        sequence_split_seed=sequence_split_seed,
        keep_rate=keep_rate,
        max_frames_per_sequence=max_frames_per_sequence,
        include_torque_proxy=include_torque_proxy,
        torque_proxy_use_fps=torque_proxy_use_fps,
        torque_proxy_filter_percentile=torque_proxy_filter_percentile,
    )
    source_sequence_to_id = {}
    source_sequence_names = {}
    for ds_name in source_dataset_names:
        npz_fnames = find_amass_motion_files(amass_dir, ds_name)
        source_sequence_names[ds_name] = [
            osp.relpath(npz_fname, amass_dir).replace('\\', '/')
            for npz_fname in npz_fnames
        ]
        source_sequence_to_id[ds_name] = {
            npz_fname: sequence_id for sequence_id, npz_fname in enumerate(npz_fnames)
        }

    def fetch_from_amass(split_name, ds_names):
        for ds_name in ds_names:
            mosh_stageII_fnames = sequence_assignments.get((split_name, ds_name), [])
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
                if save_source_metadata:
                    data['source_dataset'] = np.full(
                        len(cdata_ids),
                        source_dataset_to_id[ds_name],
                        dtype=np.int64,
                    )
                    data['source_sequence'] = np.full(
                        len(cdata_ids),
                        source_sequence_to_id[ds_name].get(npz_fname, -1),
                        dtype=np.int64,
                    )
                if include_torque_proxy:
                    selected_torque_proxy = torque_proxy[cdata_ids]
                    if torque_proxy_log1p:
                        # Compress the long-tailed proxy scale before training normalization.
                        selected_torque_proxy = np.log1p(np.maximum(selected_torque_proxy, 0.0))
                    data['torque_proxy'] = selected_torque_proxy.astype(np.float32)
                ds_frame_count += len(cdata_ids)
                yield data

    for split_name, ds_names in amass_splits.items():
        if dataset_exists(vposer_dataset_dir, split_names=[split_name], data_fields=required_data_fields): continue
        logger(f'Preparing VPoser data for split {split_name}')

        split_data_fields = {}
        for data in fetch_from_amass(split_name, ds_names):
            for k in data.keys():
                if k not in split_data_fields: split_data_fields[k] = []
                split_data_fields[k].append(data[k])

        split_count = 0
        for k, v in split_data_fields.items():
            outpath = makepath(vposer_dataset_dir, split_name, '{}.pt'.format(k), isfile=True)
            v = np.concatenate(v)
            split_count = len(v)
            torch.save(torch.tensor(v), outpath)

        logger(
            f'{split_count} datapoints dumped for split {split_name}. ds_meta_pklpath: {osp.join(vposer_dataset_dir, split_name)}')

    if save_source_metadata:
        source_names_path = makepath(vposer_dataset_dir, 'source_dataset_names.json', isfile=True)
        with open(source_names_path, 'w') as source_names_file:
            json.dump(source_dataset_names, source_names_file, indent=2)
        logger(f'Dumped source dataset name map at {source_names_path}')
        sequence_names_path = makepath(vposer_dataset_dir, 'source_sequence_names.json', isfile=True)
        with open(sequence_names_path, 'w') as sequence_names_file:
            json.dump(source_sequence_names, sequence_names_file, indent=2)
        logger(f'Dumped source sequence name map at {sequence_names_path}')

    Configer(**{
        'amass_splits': amass_splits.toDict(),
        'amass_dir': amass_dir,
    }).dump_settings(makepath(vposer_dataset_dir, 'settings.ini', isfile=True))

    logger(f'Dumped final pytorch dataset at {vposer_dataset_dir}')
