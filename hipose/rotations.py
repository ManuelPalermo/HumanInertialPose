
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform.rotation import Slerp


'''
TODO: extend all quat operations to work on NDimensional arrays 
    -(perform operations over the last axis)
    -can use (inpt)flaten->op->reshape(out) sequence of operations
'''


def convert_quat_xyzw_to_wxyz(quat):
    """Convert quaternion from xyzw to wxyz representation."""
    return quat[..., [3, 0, 1, 2]]


def convert_quat_wxyz_to_xyzw(quat):
    """Convert quaternion from wxyz to xyzw representation."""
    return quat[..., [1, 2, 3, 0]]


def quat_diff(q1, q2, quat_scalar_last=False):
    """Compute the rotation of quaternions arrays q1 relative to q2.

    Args:
        q1(np.ndarray[Nx4]): quaternion arrays.
        q2(np.ndarray[Nx4]): quaternion arrays.
        quat_scalar_last(bool): quaternion convention to use.

    Returns:
        (np.ndarray[Nx4]): relative quaternion rotation arrays.

    """
    if quat_scalar_last:
        return (R.from_quat(q2).inv() * R.from_quat(q1)).as_quat()
    else:
        q1 = convert_quat_wxyz_to_xyzw(q1)
        q2 = convert_quat_wxyz_to_xyzw(q2)
        q = (R.from_quat(q2).inv() * R.from_quat(q1)).as_quat()
        return convert_quat_xyzw_to_wxyz(q)


def quat_mult(q1, q2, quat_scalar_last=False):
    """Compute rotation resulting from the rotation of quat1 by quat2.

    Args:
        q1(np.ndarray[Nx4]):  quaternion arrays.
        q2(np.ndarray[Nx4]):  quaternion arrays.
        quat_scalar_last(bool): quaternion convention to use.

    Returns:
        (np.ndarray[Nx4]): resulting rotation quaternion arrays.

    """
    if quat_scalar_last:
        return (R.from_quat(q1) * R.from_quat(q2)).as_quat()
    else:
        q1 = convert_quat_wxyz_to_xyzw(q1)
        q2 = convert_quat_wxyz_to_xyzw(q2)
        q = (R.from_quat(q1) * R.from_quat(q2)).as_quat()
        return convert_quat_xyzw_to_wxyz(q)


def quat_inv(quat, quat_scalar_last=False):
    """Inverts rotation represented by quaternion."""
    if quat_scalar_last:
        return R.from_quat(quat).inv().as_quat()
    else:
        quat = convert_quat_wxyz_to_xyzw(quat)
        q = R.from_quat(quat).inv().as_quat()
        return convert_quat_xyzw_to_wxyz(q)


def quat_avg(quat, quat_scalar_last=False):
    """Averages the rotations represented by quaternions."""
    if quat_scalar_last:
        return R.from_quat(quat).mean()
    else:
        quat = convert_quat_wxyz_to_xyzw(quat)
        q = R.from_quat(quat).mean().as_quat()
        return convert_quat_xyzw_to_wxyz(q)


def quat_random(num=None, quat_scalar_last=False):
    """Creates random quaternion rotations."""
    if quat_scalar_last:
        return R.random(num=num).as_quat()
    else:
        q = R.random(num=num).as_quat()
        return convert_quat_xyzw_to_wxyz(q)


def vec_rotate(vec, quat, quat_scalar_last=False):
    """Rotates a vector by the rotation represented by a quaternion."""
    if quat_scalar_last:
        return R.from_quat(quat).apply(vec)
    else:
        quat = convert_quat_wxyz_to_xyzw(quat)
        return R.from_quat(quat).apply(vec)


def quat_slerp(q1, q2, orig_points, target_points, axis=0, quat_scalar_last=False):
    if quat_scalar_last:
        orig_quats = R.from_quat(np.concatenate([q1, q2], axis=axis))
        slerp = Slerp(orig_points, orig_quats)
        return slerp(target_points).as_quat()
    else:
        q1 = convert_quat_wxyz_to_xyzw(q1)
        q2 = convert_quat_wxyz_to_xyzw(q2)
        orig_quats = R.from_quat(np.concatenate([q1, q2], axis=axis))
        slerp = Slerp(orig_points, orig_quats)
        new_quats = slerp(target_points).as_quat()
        return convert_quat_xyzw_to_wxyz(new_quats)


def convert_rotmat_to_quat(rotmat, quat_scalar_last=False):
    """Convert rotation matrix to quaternion."""
    if quat_scalar_last:
        return R.from_matrix(rotmat).as_quat()
    else:
        return convert_quat_xyzw_to_wxyz(
                R.from_matrix(rotmat).as_quat())


def convert_quat_to_rotmat(quat, quat_scalar_last=False):
    """Convert quaternion rotation to rotation matrix."""
    if quat_scalar_last:
        return R.from_quat(quat).as_matrix()
    else:
        return R.from_quat(
                convert_quat_wxyz_to_xyzw(quat)
        ).as_matrix()


def convert_euler_to_quat(euler, seq="xyz", quat_scalar_last=False):
    """Convert euler angles to quaternion."""
    if quat_scalar_last:
        return R.from_euler(seq, euler).as_quat()
    else:
        return convert_quat_xyzw_to_wxyz(
                R.from_euler(seq, euler).as_quat())


def convert_quat_to_euler(quat, seq="xyz", degrees=False, quat_scalar_last=False):
    """Convert quaternion rotation to euler angles."""
    if quat_scalar_last:
        return R.from_quat(quat).as_euler(seq, degrees=degrees)
    else:
        return R.from_quat(
                convert_quat_wxyz_to_xyzw(quat)
        ).as_euler(seq, degrees=degrees)


def convert_quat_to_euler_continuous(quat, seq="xyz", degrees=False,
                                     quat_scalar_last=False):
    """Convert quaternion rotation to euler removing discontinuities."""
    if not quat_scalar_last:
        quat = convert_quat_wxyz_to_xyzw(quat)

    oshape = tuple([*quat.shape[:-1], 3])
    euler_ori = R.from_quat(quat.reshape(-1, 4)).as_euler(seq).reshape(*oshape)

    euler_ori = np.unwrap(euler_ori, axis=0)    # remove discontinuities

    if degrees:
        euler_ori = np.rad2deg(euler_ori)
    return euler_ori


def qcip(quat1, quat2, keepdims=True):
    """
    Cosine of inner quaternion products [0, pi/2].

    paper (Φ3): https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    Based on: https://github.com/Mayitzin/ahrs/blob/master/ahrs/utils/metrics.py#L308
    """
    quat1 = quat1 / np.linalg.norm(quat1, axis=-1, keepdims=True)
    quat2 = quat2 / np.linalg.norm(quat2, axis=-1, keepdims=True)
    d = np.abs(np.sum(quat1 * quat2, axis=-1, keepdims=keepdims))
    return np.arccos(d)


def qad(quat1, quat2, eps=1e-6, keepdims=True):
    """Returns the angular distance between 2 quaternions [0, pi].

    paper (Φ6): https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    Based on: https://github.com/Mayitzin/ahrs/blob/master/ahrs/utils/metrics.py#L341
    """
    quat1 = quat1 / np.linalg.norm(quat1, axis=-1, keepdims=True)
    quat2 = quat2 / np.linalg.norm(quat2, axis=-1, keepdims=True)
    d = np.clip(
            2.0 * np.sum(quat1 * quat2, axis=-1, keepdims=keepdims) ** 2 - 1.0,
            -1 + eps, 1 - eps)
    return np.arccos(d)


def wrap_angle(angles, degrees=False):
    """
    Wraps angles between [-pi, pi].
    """
    ang = 180 if degrees else np.pi
    return ((angles + ang) % (2.0 * ang)) - ang
