import numpy as np
from scipy.constants.constants import g

from ahrs.filters import AngularRate, Mahony, Madgwick, EKF
from ahrs.common.orientation import acc2q, ecompass

from hipose.rotations import quat_mult, quat_inv, vec_rotate, \
    convert_quat_to_euler, convert_euler_to_quat, quat_avg


def rotate_vectors(*vector_data, rotation, inv=False):
    """Util function to apply the same rotation to multiple vectors.

    Args:
        vector_data([Iterable[np.ndarray[..., 3]]]): one or multiple
            vectors of data to rotate.
        rotation(np.ndarray[..., 4]): quaternion rotation to
            be applied.
        inv(bool): if inverse rotation should be applied.

    Returns:
        (tuple[np.ndarray[..., 3]]): rotated vector data.

    """
    assert len(vector_data) > 0
    rotated_data = []
    for vec in vector_data:
        if len(vec.shape) <= 2:  # scipy can only handle 2 dims
            qrot = (quat_inv(rotation) if inv else rotation)
            rotated_data.append(vec_rotate(vec, qrot))

        else:  # more than 2 dims (flatten extra dims)
            oshape = vec.shape
            vec = vec.reshape(-1, 3)

            # handle only 1 rotation for all data or individual rotation for each sample
            qrot = (rotation
                    if len(rotation.shape) == 1
                    else rotation.reshape(-1, 4))
            qrot = (quat_inv(qrot) if inv else qrot)
            rotated_data.append(vec_rotate(vec, qrot).reshape(*oshape))

    return tuple(rotated_data)


def compute_imu_orientation(acc_data, gyr_data, mag_data=None,
                            algorithm="madgwick", freq=100.0,
                            q0=None, n_init_samples=10):
    """Compute the orientation of IMU sensor using fusion filter.

    Computes the orientation of an imu sensor from the raw inertial
    data using fusion filters. Assumes imu data is in NWU
    reference frame. Output data will also be in NWU reference frame.

    Args:
        acc_data(np.ndarray[Nx3]): imu accelerometer data.
        gyr_data(np.ndarray[Nx3]): imu gyroscope data.
        mag_data(None|np.ndarray[Nx3]): imu magnetometer data. If None,
            algorithms will use versions without magnetometer data
            if possible.
        algorithm(str): algorithm to use to fuse sensor data. Can be
            one of ["integral", "mahony", "madgwick", "ekf"]. Defaults
             to madgwick.
        freq(float): data acquisition frequency.
        q0(None|np.ndarray[4x]): initial orientation quaternion. If
            None then a default orientation is computed using
            the first data sample. q0 should be in NWU referential.
        n_init_samples(int): number of initial samples to initialize
            q0 if its not passed directly as argument.

    Returns:
        (np.ndarray[Nx4]): orientation array of unit quaternion.

    """
    algorithm = algorithm.lower()

    # get initial orientation acc/mag references if not given
    if q0 is not None:
        # convert from NWU to NED from ahrs library
        q0 = quat_mult(convert_euler_to_quat([np.pi, 0, 0], seq="xyz"), q0)
        q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
    else:
        q0 = (ecompass(np.mean(acc_data[:n_init_samples], axis=0),
                       np.mean(mag_data[:n_init_samples], axis=0),
                       representation="quaternion", frame="NED")
              if mag_data is not None
              else acc2q(np.mean(acc_data[:n_init_samples], axis=0)))
        q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)

    if algorithm == "madgwick":
        orient = Madgwick(acc=acc_data, gyr=gyr_data,
                          mag=mag_data, frequency=freq, q0=q0).Q

    elif algorithm == "mahony":
        orient = Mahony(acc=acc_data, gyr=gyr_data,
                        mag=mag_data, frequency=freq, q0=q0).Q

    elif algorithm == "ekf":
        if mag_data is not None:
            from ahrs.utils.wmm import WMM
            # magnetic info for braga (change accordingly)
            wmm = WMM(latitude=41.5517, longitude=8.4229, height=0.20)
            mref = np.array([wmm.X, wmm.Y, wmm.Z])
        else:
            mref = None
        # (ENU gives expected results for NED frame (bug in ahrs library)?)
        orient = EKF(acc=acc_data, gyr=gyr_data,
                     mag=mag_data, frequency=freq, q0=q0,
                     frame="ENU", magnetic_ref=mref).Q

    elif algorithm == "integral":
        orient = AngularRate(gyr=gyr_data, frequency=freq, q0=q0).Q

    else:
        raise NotImplementedError(
                f"Chosen orientation algorithm not implemented '{algorithm}'! "
                "Choose one of [integral, mahony, madgwick, ekf]")

    if algorithm not in ("integral", ):
        # rotate orientation from NED (computed by library) to NWU (used by us)
        # could probably also rotate raw data before? (but seems to work like this)
        orient = quat_mult(
                convert_euler_to_quat([np.pi, 0, 0], seq="xyz"),
                orient
        )
    return orient


def relative_orientations(orient_data, ref_quat):
    """Transform multiple orientations relative to a reference one.

    Args:
        orient_data(np.ndarray[...xIx4]): orientation quaternion.
        ref_quat(np.ndarray[...1x4]): reference quaternion orientation.

    Returns:
        (np.ndarray[...xIx4]): Array of quaternions with relative
            orientations for each IMU.

    """
    in_shape = np.shape(orient_data)
    assert len(in_shape) > 1
    orient_data = orient_data.reshape((-1, in_shape[-1]))
    return quat_mult(quat_inv(ref_quat), orient_data).reshape(in_shape)


def static_sensor_to_segment_calibration(acc_data, mag_data=None):
    """Compute sensor to segment offset from given data.

    Args:
        acc_data(np.ndarray[Nx3]): imu accelerometer data.
        mag_data(None|np.ndarray[Nx3]): imu magnetometer data. If None,
            algorithm will use versions without magnetometer data.

    Returns:
        (np.ndarray[Nx4]): rotation offsets as quaternions.

    """
    if mag_data is not None:
        # normalize data and compute sensor orientation using TRIAD method
        # (rotation which aligns sensor ref to world ref - gravity / north-pole)
        acc_data /= np.linalg.norm(acc_data, axis=-1, keepdims=True)
        mag_data /= np.linalg.norm(mag_data, axis=-1, keepdims=True)
        s2s_offsets = np.array([ecompass(a, m, frame="NED",
                                         representation="quaternion")
                                for a, m in zip(acc_data, mag_data)])

        # rotate orientation from NED (computed by library) to NWU (used here)
        s2s_offsets = quat_mult(
                convert_euler_to_quat([np.pi, 0, 0], seq="XYZ"),
                s2s_offsets
        )

    else:
        acc_data /= np.linalg.norm(acc_data, axis=-1, keepdims=True)
        s2s_offsets = np.array([acc2q(a) for a in acc_data])

        # rotate orientation from NED (computed by library) to NWU (used by here)
        s2s_offsets = quat_mult(
                convert_euler_to_quat([np.pi, 0, 0], seq="XYZ"),
                s2s_offsets
        )

    s2s_offsets = quat_inv(s2s_offsets)
    return s2s_offsets / np.linalg.norm(s2s_offsets, axis=-1, keepdims=True)


def dynamic_optim_sensor_to_segment_calibration(dyn_calib_traj_acc, dyn_calib_traj_gyr,
                                                ref_traj_acc, ref_traj_gyr,
                                                initial_s2s_guess=None,
                                                n_warmup=5000, n_generations=100,
                                                acc_weight=1., gyr_weight=1.,
                                                smooth_window=201, verbose=False):
    """
    Sensor to segment calibration through CMA-ES optimization.

    Sensor to segment calibration by minimizing the measured error
    between a dynamic trajectory with respect to a reference trajectory,
    based on accelerometer and gyroscope data. Longer trajectories with
    high variability of movement and slow dynamics seem to work best.

    """
    from .metrics import rmse
    from .rotations import quat_random

    try:
        from cmaes import CMA, get_warm_start_mgd
    except ModuleNotFoundError:
        print("cmaes library needs to be installed. Use 'pip install cmaes'. ")

    assert (dyn_calib_traj_acc.shape == dyn_calib_traj_gyr.shape
            == ref_traj_acc.shape == ref_traj_gyr.shape)

    if smooth_window is not None:
        from scipy.signal import savgol_filter
        dyn_calib_traj_acc = savgol_filter(dyn_calib_traj_acc, window_length=smooth_window, polyorder=2, axis=0)
        dyn_calib_traj_gyr = savgol_filter(dyn_calib_traj_gyr, window_length=smooth_window, polyorder=2, axis=0)
        ref_traj_acc = savgol_filter(ref_traj_acc, window_length=smooth_window, polyorder=2, axis=0)
        ref_traj_gyr = savgol_filter(ref_traj_gyr, window_length=smooth_window, polyorder=2, axis=0)

    dyn_calib_traj_acc /= np.linalg.norm(dyn_calib_traj_acc, axis=-1, keepdims=True)
    ref_traj_acc /= np.linalg.norm(ref_traj_acc, axis=-1, keepdims=True)

    # distance function: quaternion angle distance
    def dist_func(s2s_offset):
        align_traj_acc, align_traj_gyr = \
            rotate_vectors(dyn_calib_traj_acc, dyn_calib_traj_gyr,
                           rotation=s2s_offset, inv=True)
        dist_acc = acc_weight * float(rmse(align_traj_acc, ref_traj_acc, reduce=True))
        dist_gyr = gyr_weight * float(rmse(align_traj_gyr, ref_traj_gyr, reduce=True))
        return dist_acc + dist_gyr

    # estimate a promising distribution from random sampling,
    # then generate parameters of the multivariate gaussian distribution.
    init_solutions = ([(initial_s2s_guess, dist_func(initial_s2s_guess))]
                      if initial_s2s_guess is not None
                      else [])

    for q0 in quat_random(num=n_warmup):
        dist = dist_func(q0)
        init_solutions.append((q0, dist))
    ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(init_solutions, gamma=0.1, alpha=0.1)
    ws_mean = ws_mean if initial_s2s_guess is None else initial_s2s_guess.copy()

    # initialize CMAES optimizer
    optim = CMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov,
                bounds=np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]))

    best_s2s = None
    best_dist = np.inf
    for g in range(n_generations):
        solutions = []
        for _ in range(optim.population_size):
            q = optim.ask()
            dist = dist_func(q)
            solutions.append((q, dist))
        optim.tell(solutions)

        if verbose and (optim.generation - 1) % (n_generations // 10) == 0:
            print(f"gen:{g}   d:{dist:.4f}   q:{np.round(q, 3)}")

        # get one of the solutions (all similar at this point)
        # and save it if its the best so far
        solv_s2s, solv_dist = solutions[0]
        if solv_dist < best_dist:
            best_s2s = solv_s2s.copy()
            best_dist = solv_dist

        if optim.should_stop():
            # (IPOP-CMA-ES) restart CMAES search with population_size * 2
            optim = CMA(mean=ws_mean, sigma=ws_sigma,
                        population_size=optim.population_size * 2)

    if verbose:
        print("\nOptimized s2s values:")
        np.set_printoptions(suppress=True)
        if initial_s2s_guess is not None:
            init_dist = dist_func(initial_s2s_guess)
            print("Init:  d:", init_dist, "euler:", convert_quat_to_euler(initial_s2s_guess, seq="XYZ", degrees=True))
        print("Pred:  d:", best_dist, "euler:", convert_quat_to_euler(best_s2s, seq="XYZ", degrees=True))
        print("------------------------------\n")
    return best_s2s / np.linalg.norm(best_s2s, axis=-1, keepdims=True)


def imus_static_calibration(acc_calib_data, gyr_calib_data, mag_calib_data,
                            manual_align_rots=None, s2s_calib_method=None):
    """Compute calibration parameters from imu data in static pose.

    acc_calib_data(np.ndarray[NxIx3]): raw accelerometer data
        containing n-pose standing calibration samples.
    gyr_calib_data(np.ndarray[NxIx3]): raw gyroscope data containing
        n-pose standing calibration samples.
    mag_calib_data(np.ndarray[NxIx3]): raw magnetometer data containing
        n-pose standing calibration samples.
    manual_align_rots(None|np.ndarray[Ix4]): quaternion rotations to
        apply to manually rotate each sensor to closer to correct
        position.
    s2s_calib_method(None|str): sensor to segment calibration method to
        apply. Can be one of: [None, "manual", "static", "static_mag"].

    Returns:
        (dict[str, np.ndarray]): dictionary containing extracted
            calibration parameters.

    """
    assert s2s_calib_method in [None, "manual", "static", "static_mag"]
    assert s2s_calib_method != "manual" or manual_align_rots is not None

    # compute sensor calibration params (in original ref)
    acc_magn = np.mean(np.linalg.norm(acc_calib_data, axis=-1), axis=0)
    gyr_bias = gyr_calib_data.mean(axis=0)
    mag_magn = np.mean(np.linalg.norm(mag_calib_data, axis=-1), axis=0)

    # compute sensor to segment calibration
    if s2s_calib_method is None:
        s2s_offsets = np.tile([1, 0, 0, 0], reps=(len(gyr_bias), 1))

    elif s2s_calib_method == "manual":
        s2s_offsets = manual_align_rots

    elif s2s_calib_method == "static":
        # add manual calibration to s2s calibration
        # if available, manually rotate IMUs data so they more closely
        # align with desired reference

        if manual_align_rots is not None:
            # need to manually broadcast rotation to match data dimensions
            _rots = np.repeat(
                    manual_align_rots.reshape((1, *manual_align_rots.shape)),
                    len(acc_calib_data),
                    axis=0)

            # rotate imu data to manually aligned ref
            acc_calib_data, gyr_calib_data, mag_calib_data = rotate_vectors(
                    acc_calib_data, gyr_calib_data, mag_calib_data,
                    rotation=_rots, inv=True)

        # calculate s2s static calibration using gravity reference
        s2s_offsets = static_sensor_to_segment_calibration(
                acc_data=np.mean(acc_calib_data, axis=0),
                mag_data=None)

        if manual_align_rots is not None:
            # add manual calibration to s2s calibration
            s2s_offsets = quat_mult(manual_align_rots, s2s_offsets)

    elif s2s_calib_method == "static_mag":
        # calculate s2s calibration using gravity and magnetic field references
        s2s_offsets = static_sensor_to_segment_calibration(
                acc_data=np.mean(acc_calib_data, axis=0),
                mag_data=np.mean(mag_calib_data, axis=0))

    return dict(acc_magn=acc_magn,
                gyr_bias=gyr_bias,
                mag_magn=mag_magn,
                s2s_offset=s2s_offsets)


def apply_imu_calibration(acc_data, gyr_data, mag_data,
                          acc_bias=None, acc_magn=None,
                          gyr_bias=None,
                          mag_hard_iron=None, mag_soft_iron=None, mag_magn=None,
                          s2s_offset=None):
    """
    Applies imu calibration parameters.

    Args:
        acc_data(np.ndarray[Nx3]): imu accelerometer data.
        gyr_data(np.ndarray[Nx3]): imu gyroscope data.
        mag_data(np.ndarray[Nx3]): imu magnetometer data.
        acc_bias(list[float, float, float], None): accelerometer XYZ
            bias params. If None, then no calibration is applied.
        acc_magn(float, None): accelerometer magnitude. Can be applied
            if "acc_bias" is not available.
        gyr_bias(list[float, float, float], None): gyroscope XYZ bias
            params. If None, then no calibration is applied.
        mag_hard_iron(list[float, float, float], None): magnetometer
            XYZ hard-iron bias params. If None, then no calibration
            is applied.
        mag_soft_iron(list[float, float, float], None): magnetometer
            XYZ soft-iron bias params. If None, then no calibration
            is applied.
        mag_magn(float, None): magnetometer magnetic field norm
            (assumes hard/soft bias removed). If None, then no
            calibration is applied.
        s2s_offset(np.ndarray[Nx4]): quaternion rotations to apply to
            IMU data for sensor to segment correction.

    Returns:
        (np.ndarray[Nx3], np.ndarray[Nx3], np.ndarray[Nx3]):
            calibrated_acc, calibrated_gyr, calibrated_mag.

    """
    # apply acc calibration if available
    if acc_bias is not None:
        acc_data = acc_data - acc_bias
    if acc_magn is not None:
        acc_data = (acc_data / acc_magn) * g

    # apply gyr calibration if available
    if gyr_bias is not None:
        gyr_data = gyr_data - gyr_bias

    # apply mag calibration if available
    if mag_hard_iron is not None:
        mag_data = mag_data - mag_hard_iron
    if mag_soft_iron is not None:
        mag_data = mag_data * mag_soft_iron
    if mag_magn is not None:
        mag_data = mag_data / mag_magn

    # apply sensor to segment offset correction
    if s2s_offset is not None:
        acc_data, gyr_data, mag_data = rotate_vectors(
                acc_data, gyr_data, mag_data,
                rotation=s2s_offset, inv=True)
    return acc_data, gyr_data, mag_data


def magnetometer_soft_hard_bias_calibration(mag_calib_data, visualize=False):
    """Computes soft and hard bias parameters from magnetometer data.

    Requires a good sampling over all 3-axis. Follows implementation from:
    https://github.com/kriswiner/MPU6050/wiki/Simple-and-Effective-Magnetometer-Calibration

    Args:
        mag_calib_data(np.ndarray[..., 3]): magnetometer data with
            sensors moving around 360º over all axis.
        visualize(bool): if data before and after calibration should
            be plotted (ex. for debugging).

    Returns:
        (np.ndarray[..., 3], np.ndarray[..., 3]): hard and soft bias
            calibration params.

    """
    from .utils import remove_outliers

    filt_mag_calib_data = remove_outliers(mag_calib_data, std=2.0)
    hxmax, hxmin = filt_mag_calib_data[..., 0].max(), filt_mag_calib_data[..., 0].min()
    hymax, hymin = filt_mag_calib_data[..., 1].max(), filt_mag_calib_data[..., 1].min()
    hzmax, hzmin = filt_mag_calib_data[..., 2].max(), filt_mag_calib_data[..., 2].min()

    # hard iron calibration
    hxb = (hxmax + hxmin) / 2.
    hyb = (hymax + hymin) / 2.
    hzb = (hzmax + hzmin) / 2.
    hard_bias = np.array([hxb, hyb, hzb])

    # simplistic soft iron calibration
    hxs = (hxmax - hxmin) / 2.
    hys = (hymax - hymin) / 2.
    hzs = (hzmax - hzmin) / 2.
    soft_bias = np.array([hxs, hys, hzs])
    soft_bias = soft_bias.mean() / soft_bias

    if visualize:
        import matplotlib.pyplot as plt
        mag_no_bias_data = (filt_mag_calib_data - hard_bias) * soft_bias

        fig, ax = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
        for i, (name, data) in enumerate(zip(["Before Calib", "After Calib"],
                                             [mag_calib_data, mag_no_bias_data])):
            ax[i].scatter(data[:, 0], data[:, 1], alpha=0.25)  # x plane
            ax[i].scatter(data[:, 0], data[:, 2], alpha=0.25)  # y plane
            ax[i].scatter(data[:, 1], data[:, 2], alpha=0.25)  # z plane
            ax[i].legend(["x", "y", "z"])
            ax[i].set_title(name)
        plt.show(block=False)
    return hard_bias, soft_bias


def remove_gravity_acceleration(acc, orient, ref_frame="sensor"):
    """
    Compute the free acceleration by removing the gravity component.
    The resulting acceleration can be returned w.r.t to the world
    (linear) or sensor reference (free).

    Args:
        acc(np.ndarray[Nx3]): imu sensor measured acceleration
        orient(np.ndarray[Nx4]): imu orientation quaternion.
        ref_frame(str): referential frame to return acceleration:
            ("sensor" or "world").
    Returns:
        (np.ndarray[Nx3]): acceleration without gravity vector

    """
    assert ref_frame in ["world", "sensor"]
    world_g_vec = np.array([0., 0., -g])
    if ref_frame == "world":
        return vec_rotate(acc, orient) - world_g_vec
    else:
        return acc - vec_rotate(world_g_vec, quat_inv(orient))


def add_gravity_acceleration(acc, orient, ref_frame="sensor"):
    """
    Add gravity vector component to sensor free acceleration. Can deal
    with free (w.r.t to the sensor frame) or linear (w.r.t to the world)
    accelerations. Results are returned in sensor reference frame.

    Args:
        acc(np.ndarray[Nx3]): imu sensor measured acceleration.
        orient(np.ndarray[Nx4]): imu orientation quaternion.
        ref_frame(str): referential frame of the input acceleration:
            ("sensor" or "world").

    Returns:
        (np.ndarray[Nx3]): sensor acceleration with gravity vector

    """
    assert ref_frame in ["world", "sensor"]
    world_g_vec = np.array([0., 0., -g])
    if ref_frame == "world":
        return vec_rotate(acc + world_g_vec, quat_inv(orient))
    else:
        return acc + vec_rotate(world_g_vec, quat_inv(orient))


def magnetometer_heading(mag, acc=None, degrees=False, frame="NWU"):
    """Calculate magnetic north heading from magnetometer data.

    Args:
        mag(np.ndarray[..., 3]): magnetometer data.
        acc(np.ndarray[..., 3]): accelerometer data. Used for
            magnetometer tilt error compensation.
        degrees(bool): if heading should be returned in degrees
            instead of rads.
        frame(str): frame to return results. Can be one of
            ["NED", "NWU"].

    Returns:
        (np.ndarray): compass headings.

    """
    assert frame in ["NED", "NWU"]

    if acc is not None:
        # compensate for magnetometer inclination error
        mag /= np.linalg.norm(mag, axis=-1, keepdims=True)
        acc /= np.linalg.norm(acc, axis=-1, keepdims=True)
        ori = np.array([
                ecompass(a, m, frame="NED", representation="quaternion")
                for a, m in zip(acc.reshape(-1, 3), mag.reshape(-1, 3))
        ])
        heading = convert_quat_to_euler(ori, seq="xyz")[..., -1]
        heading = heading.reshape(*mag.shape[:-1], 1)
    else:
        mag /= np.linalg.norm(mag, axis=-1, keepdims=True)
        heading = np.arctan2(mag[..., 1], mag[..., 0])

    # convert output to desired format
    if frame == "NWU":
        heading = -heading
    if degrees:
        heading = np.rad2deg(heading)
    return heading
