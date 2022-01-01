
import numpy as np

from hipose.rotations import convert_euler_to_quat, quat_mult, quat_slerp


# xsens to xsens upperbody
def map_segs_xsens2xsensupper(xsens_segs):
    return xsens_segs[..., [0, 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 10,
                            11, 12, 13, 14], :]


def _map_kpts_xsens2xsensupper(xsens_kpts):
    raise NotImplementedError


# xsens to ergowear
def map_segs_xsens2ergowear(xsens_segs):
    return xsens_segs[..., [0, 4, 5,
                            8, 9, 10,
                            12, 13, 14], :]


def map_kpts_xsens2ergowear(xsens_kpts):
    return xsens_kpts[..., [0, 3, 5, 8,
                            9, 10, 12, 13,
                            14, 23, 24, 25], :]


# xsens to mtwawinda
def map_segs_xsens2mtwawinda(xsens_segs):
    return xsens_segs[..., [0, 4, 6,
                            7, 8, 9, 10,
                            11, 12, 13, 14,
                            15, 16, 17,
                            19, 20, 21], :]


def _map_kpts_xsens2mtwawinda(xsens_kpts):
    raise NotImplementedError


# mtwawinda to ergowear
def map_segs_mtwawinda2ergowear(mtwawinda_segs):
    return mtwawinda_segs[..., [0, 1, 2,
                                4, 5, 6,
                                8, 9, 10], :]


def _map_kpts_mtwawinda2ergowear(mtwawinda_kpts):
    raise NotImplementedError


# mtwawinda to xsens
def map_segs_mtwawinda2xsens(mtwawinda_segs):
    spine_segs = quat_slerp(q1=mtwawinda_segs[..., [0], :],
                            q2=mtwawinda_segs[..., [1], :],
                            orig_points=[0, 4],
                            target_points=[0, 0.75, 1.5, 3.25, 4],
                            axis=0)

    return np.concatenate([spine_segs,                                  # spine
                           mtwawinda_segs[..., [2, 2], :],              # neck
                           mtwawinda_segs[..., [3, 4, 5, 6,             # r_arm
                                                7, 8, 9, 10,            # l_arm
                                                11, 12, 13, 13,         # r_leg
                                                14, 15, 16, 16], :],    # l_leg
                          ], axis=-2)


def _map_kpts_mtwawinda2xsens(mtwawinda_kpts):
    raise NotImplementedError


# ergowear to xsens-upperbody
def map_segs_ergowear2xsensupper(ergo_segs):
    seg_r_scapula = quat_mult(
            ergo_segs[..., [1], :],
            convert_euler_to_quat([0, 0, 0], seq="XYZ"))

    seg_l_scapula = quat_mult(
            ergo_segs[..., [1], :],
            convert_euler_to_quat([0, 0, 0], seq="XYZ"))

    # interpolate spine points
    spine_segs = quat_slerp(q1=ergo_segs[..., [0], :],
                            q2=ergo_segs[..., [1], :],
                            orig_points=[0, 4],
                            target_points=[0, 0.75, 1.5, 3.25, 4],
                            axis=0)

    return np.concatenate([spine_segs,                      # spine
                           ergo_segs[..., [2, 2], :],       # neck
                           seg_r_scapula,
                           ergo_segs[..., [3, 4, 5], :],    # r_arm
                           seg_l_scapula,
                           ergo_segs[..., [6, 7, 8], :]],   # l_arm
                          axis=-2)


def _map_kpts_ergowear2xsensupper(ergo_kpts):
    raise NotImplementedError


# ergowear to mtwawinda-upperbody
def map_segs_ergowear2mtwawindaupper(ergo_segs):
    seg_r_scapula = quat_mult(
            ergo_segs[..., [1], :],
            convert_euler_to_quat([0, 0, 0], seq="XYZ"))

    seg_l_scapula = quat_mult(
            ergo_segs[..., [1], :],
            convert_euler_to_quat([0, 0, 0], seq="XYZ"))

    return np.concatenate([ergo_segs[..., [0, 1, 2], :],
                           seg_r_scapula,
                           ergo_segs[..., [3, 4, 5], :],
                           seg_l_scapula,
                           ergo_segs[..., [6, 7, 8], :]],
                          axis=-2)

def _map_kpts_ergowear2mtwawindaupper(ergo_kpts):
    raise NotImplementedError
