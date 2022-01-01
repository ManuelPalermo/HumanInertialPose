
import os
import glob

import numpy as np

from scipy import interpolate
from scipy.spatial.transform import Rotation as R, RotationSpline

from copy import deepcopy
from collections import defaultdict

# nested dict replacement
from .rotations import convert_quat_wxyz_to_xyzw, \
    convert_quat_xyzw_to_wxyz, quat_mult, convert_quat_to_euler,\
    convert_euler_to_quat, quat_inv, vec_rotate


nested_dict = lambda: defaultdict(nested_dict)

def convert_nestedddict_to_regular(d):
    """
    Converts nested defaultdict object to regular python nested dicts.
    """
    if isinstance(d, defaultdict):
        d = {k: convert_nestedddict_to_regular(v) for k, v in d.items()}
    return d


def truncate_dict_of_arrays(dict_array, s_idx=0, e_idx=999999999999999,
                            inplace=False):
    """Truncate arrays inside a dictionary to desired start/end idx."""
    if not inplace:
        dict_array = deepcopy(dict_array)

    for k, v in dict_array.items():
        if isinstance(v, (np.ndarray, list)):
            dict_array[k] = v[s_idx:e_idx]
    return dict_array


def select_idx_dict_of_arrays(dict_array, axis_idx_dict,
                              inplace=False, ignore_idx_errors=True):
    """Selects indexes in desired axis in arrays inside a dictionary.

    Args:
        dict_array(dict[str,np.ndarray): dictionary with
            arrays or tensors.
        axis_idx_dict(dict[int,list[int]]): dictionary with indexes to
            select for each axis as {axis:indexes}.
        inplace(bool): if changes should be done on input arrays.
        ignore_idx_errors(bool): if errors on arrays with missing
            dimensions should be ignored.

    Return:
        dict[str,np.ndarray]:

    """
    out_dict_array = (dict_array if inplace else dict())
    for k, v in dict_array.items():
        ix = [axis_idx_dict.get(dim, slice(None)) for dim in range(v.ndim)]
        if isinstance(v, (np.ndarray)):
            try:
                out_dict_array[k] = v[tuple(ix)]
            except IndexError as e:
                if not ignore_idx_errors:
                    raise e
                else:
                    out_dict_array[k] = v
    return out_dict_array


def find_resource_path(path, max_up=5):
    """Recursively looks for files on the current directory and parents.

    Args:
        path(str): base path or files name of resource to search.
        max_up(int): max parent directory from which to recurse.

    Returns:
        (str): path to found resource or None if it was not found.

    """
    for i in range(max_up):
        pmatch = glob.glob(os.path.join("**/", path), recursive=True)
        if not pmatch:
            path = "../" + path
        else:
            path = pmatch[0]
            return path
    return None


def resample_data_frequency_factor(data, factor, axis=0, method="cubic"):
    """
    Resample the data by the desired factor (assumes uniform sampling).

    Args:
        data(np.ndarray): data array to resample.
        factor(float): factor to resample the data. if factor<1.0, decimation
            is performed,  otherwise, interpolation is performed.
        axis(int): index of the axis to interpolate data along.
        method(str): method to use for resampling the data. Defaults to
            cubic spline. When resampling quaternion rotations,
            use "slerp".

    Returns:
        (np.ndarray): The resampled data array.

    """
    x = np.arange(data.shape[axis])
    x_new = np.linspace(0, data.shape[axis] - 1, round(data.shape[axis] * factor), endpoint=True)

    return resample_data_frequency(data, orig_t=x, target_t=x_new,
                                   axis=axis, method=method)


def resample_data_frequency(data, orig_t, target_t, axis=0, method="cubic"):
    """
    Resample the data from original sampling to target.

    Args:
        data(np.ndarray): data array to resample.
        orig_t(np.ndarray): original timestamps for each data point.
        target_t(np.ndarray): target timestamps for resampled points.
        axis(int): index of the axis to interpolate data along.
        method(str): method to use for resampling the data. Defaults to
            cubic spline. When resampling quaternion rotations,
            use "slerp".

    Returns:
        (np.ndarray): The resampled data array.

    """
    if method == "slerp":
        assert axis == 0, "Spherical Rotation Spline only works when axis=0"
        data = convert_quat_wxyz_to_xyzw(data)  # convert quats to scalar_last for scipy
        if len(np.shape(data)) == 3:  # multiple segments
            sampled_data = np.array(
                    [RotationSpline(orig_t, R.from_quat(ori))(target_t).as_quat()
                     for ori in data.transpose((1, 0, 2))
                     ]).transpose((1, 0, 2))
        else:
            sampled_data = RotationSpline(orig_t, R.from_quat(data))(target_t).as_quat()
        return convert_quat_xyzw_to_wxyz(sampled_data)  # convert quats back to scalar_first
    else:
        return interpolate.interp1d(orig_t, data, kind=method, axis=axis,
                                    bounds_error=False, fill_value="extrapolate")(target_t)


def find_low_variability_sections(data, threshold, window_size=21,
                                  axis=0, thresh_method="max"):
    """
    Find sections of data which contain low variability.

    Args:
        data(np.ndarray): data to search.
        window_size(int): window size to average data.
        threshold(float): threshold value to consider low_variability
        axis(int): axis on which to search.
        thresh_method(str): channel reduction method to compare with
            threshold. One of ["mean", "min", "max"]

    Returns:
        (list[int]): indexes of samples with low variability over
            the desired axis

    """
    from scipy.signal import convolve
    from scipy.signal.windows import gaussian

    reduce_dims = list(np.arange(len(data.shape)))
    reduce_dims.remove(axis)
    reduce_dims = tuple(reduce_dims)

    # apply gaussian smoothing over the axis
    kernel = np.expand_dims(gaussian(window_size, std=window_size / 8), axis=reduce_dims)
    smooth_signal = convolve(data, kernel / kernel.sum(), mode="same")

    # calculate ||pointwise derivatives||
    diff = np.abs(np.diff(smooth_signal, axis=0))
    diff[-window_size:window_size, ...] = 0

    # take (mean, min, max) variability over all channels
    if thresh_method == "mean":
        avg_diff = np.mean(diff, axis=reduce_dims)
    elif thresh_method == "min":
        avg_diff = np.min(diff, axis=reduce_dims)
    elif thresh_method == "max":
        avg_diff = np.max(diff, axis=reduce_dims)
    else:
        raise NotImplementedError

    return list(np.where(avg_diff < threshold)[0])


def remove_outliers(data, std=2.0):
    """Statistical outlier removal for each axis independently.

    Args:
        data(np.ndarray): data from which to remove outliers.
        std(float): stdev for vaues to be considered outliers.

    Returns:
        (np.ndarray): data with outliers removed

    """
    assert len(data.shape) == 2
    num, dim = data.shape
    inliers = list(range(num))
    outliers = []
    for ax in range(dim):
        diff = np.append(0., np.diff(data[..., ax]))
        ax_outliers = np.where(np.abs(diff) > np.abs(np.mean(diff)) + (std * np.std(diff)))[0]
        outliers.extend(ax_outliers)

    inliers = np.array(list(set(inliers) - set(outliers)))
    return data[inliers]


def reset_skeleton_position(pos_args_list, pos_ref, axis2reset=(True, True, False)):
    """
    Resets position of data by removing a reference position.

    Args:
        pos_args_list(list[np.ndarray]): list with arrays of 3d points
            from which to remove the reference.
        pos_ref (np.ndarray[3x]): reference position.
        axis2reset (tuple[bool]): operation mask. Only axis (xyz) with
            True will be reset.

    Returns:
        (list[np.ndarray]): 3d arrays with reset position.

    """
    return [(p - pos_ref * axis2reset) for p in pos_args_list]


def reset_skeleton_orientation(rot_ref, orient_arg_list=(),
                               pos_args_list=(), vec_args_list=(),
                               axis2reset=(False, False, True)):
    """
    Reset orientation by removing a reference rotation.

    Args:
        rot_ref(np.ndarray): reference rotation.
        orient_arg_list(tuple[np.ndarray]): list with arrays of
            quaternion orientations.
        pos_args_list(tuple[np.ndarray]): list with arrays of 3d
            positions around a center point (assumed to be
            index 0 - root).
        vec_args_list(tuple[np.ndarray]): list with arrays of 3d
            vectors.
        axis2reset(tuple[bool, bool, bool]): operation mask. Only axis
            (xyz) with True will be reset.

    Returns:
        (list[np.ndarray], list[np.ndarray], list[np.ndarray]):
            orientation and position arrays with reset orientations.

    """
    # reset heading (rot over z-axis, so that subject is facing forward)
    inv_init_ori = quat_inv(
            convert_euler_to_quat(
                    convert_quat_to_euler(rot_ref, seq="xyz") * axis2reset,
                    seq="xyz",
            )
    )

    reset_ori_data = []
    for ori in orient_arg_list:
        ori_shape = ori.shape
        ori = quat_mult(inv_init_ori, ori.reshape(-1, 4)).reshape(*ori_shape)
        reset_ori_data.append(ori)

    reset_pos_data = []
    for pos in pos_args_list:
        pos_shape = pos.shape

        # center rotation to origin before rotating
        init_pos = pos[0, 0]
        pos = pos - init_pos

        # apply rotation
        pos = vec_rotate(pos.reshape(-1, 3), inv_init_ori).reshape(*pos_shape)

        # restore position to original
        pos = pos + init_pos
        reset_pos_data.append(pos)

    reset_vec_data = []
    for vec in vec_args_list:
        vec_shape = vec.shape
        vec = vec_rotate(vec.reshape(-1, 3), inv_init_ori).reshape(*vec_shape)
        reset_vec_data.append(vec)

    return reset_ori_data, reset_pos_data, reset_vec_data


def apply_procrustes_alignment(pred, target):
    """
    Applies procrustes alignment to find closest fit from "pred" to "target" data.

    Args:
        pred(np.ndarray[Nx3]): source array to be aligned.
        target(np.ndarray[Nx3]): target array for alignment.

    Returns:
        predicted_aligned - Procrustes aligned data

    """
    pred = pred[np.newaxis, ...]
    target = target[np.newaxis, ...]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(pred, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = pred - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(pred, R) + t
    return np.squeeze(predicted_aligned)
