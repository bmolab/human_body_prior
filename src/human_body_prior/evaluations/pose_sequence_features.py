"""Per-frame feature calculators for ordered pose sequences."""

import torch
import numpy as np


def pose_speed_feature(pose_body):
    """Pose-space frame difference ||pose[t] - pose[t-1]||."""
    pose_body = pose_body.view(pose_body.shape[0], -1)
    speed = torch.zeros(pose_body.shape[0], dtype=pose_body.dtype)
    speed[1:] = torch.linalg.norm(pose_body[1:] - pose_body[:-1], dim=-1)
    return speed.numpy()


def pose_acceleration_feature(pose_body):
    """Change in pose speed"""
    speed = torch.tensor(pose_speed_feature(pose_body), dtype=pose_body.dtype)
    acceleration = torch.zeros_like(speed)
    acceleration[1:] = torch.abs(speed[1:] - speed[:-1])
    return acceleration.numpy()


def pose_magnitude_feature(pose_body):
    """Average axis-angle magnitude across the 21 VPoser body joints."""
    pose_body = pose_body.view(pose_body.shape[0], 21, 3)
    return torch.linalg.norm(pose_body, dim=-1).mean(dim=-1).numpy()


#dummmy torque data placefolder
def torque_proxy_feature(pose_body):
    """Exploratory only, not a physical inverse-dynamics torque."""
    speed = pose_speed_feature(pose_body)
    acceleration = pose_acceleration_feature(pose_body)
    magnitude = pose_magnitude_feature(pose_body)
    return magnitude * (speed + acceleration)


FEATURES = {
    "pose_speed": pose_speed_feature,
    "pose_acceleration": pose_acceleration_feature,
    "pose_magnitude": pose_magnitude_feature,
    "torque_proxy": torque_proxy_feature,
}


def load_external_feature(feature_file, feature_key=None):
    """Load a per-frame feature from .npy or .npz.

    Supported shapes:
        [T]       -> one scalar per frame
        [T, J]    -> reduced by mean absolute value over joints
        [T, J, D] -> reduced by vector norm, then mean over joints

    This is the intended hook for future real torque data.
    """
    if feature_file.endswith(".npy"):
        values = np.load(feature_file)
    elif feature_file.endswith(".npz"):
        data = np.load(feature_file)
        key = feature_key
        if key is None:
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError("Feature .npz has multiple arrays; pass --feature-key. Keys: {}".format(keys))
            key = keys[0]
        values = data[key]
    else:
        raise ValueError("External features must be .npy or .npz: {}".format(feature_file))

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.mean(np.abs(values), axis=1)
    if values.ndim == 3:
        return np.linalg.norm(values, axis=-1).mean(axis=1)
    raise ValueError("Unsupported feature shape {} in {}".format(values.shape, feature_file))


def feature_names():
    """Return supported feature names for the CLI."""
    return sorted(FEATURES.keys())


def compute_feature(pose_body, feature_name):
    """Dispatch a feature name to its per-frame feature array."""
    if feature_name not in FEATURES:
        raise ValueError("Unknown feature: {}".format(feature_name))
    return FEATURES[feature_name](pose_body)
