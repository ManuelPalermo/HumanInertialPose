import os
import glob

import numpy as np

from hipose.rotations import convert_quat_to_euler_continuous, \
    quat_avg, quat_mult, vec_rotate, convert_euler_to_quat, quat_inv

from hipose.imu import add_gravity_acceleration

from hipose.utils import reset_skeleton_orientation, reset_skeleton_position


xsens_joint_names = (
        "pelvis", "l5", "l3", "t12", "t8",                                                  # 0-4
        "neck", "head", "right_shoulder", "right_upper_arm", "right_forearm",               # 5-9
        "right_hand", "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",       # 10-14
        "right_upper_leg", "right_lower_leg", "right_foot", "right_toe", "left_upper_leg",  # 15-19
        "left_lower_leg", "left_foot", "left_toe"                                           # 20-22
)

# convert segment angles from default t-pose relative to n-pose relative
convert_to_rel_npose_angles = \
    convert_euler_to_quat(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],

             [0, 0, 0],
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0],

             [0, 0, 0],
             [np.pi / 2, 0, 0],
             [np.pi / 2, 0, 0],
             [np.pi / 2, 0, 0],

             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],

             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             ],
            seq="XYZ")


def extract_xsens_analyse_raw_data(xsens_trial_path):
    """
    Extracts data from Xsens .xlsx file.

    Args:
        xsens_file_path(str): path to the trial directory containing
            xsens data.

    Returns:
        (dict): extracted xsens raw data.

    """
    xsens_files = glob.glob(xsens_trial_path + "/*.xlsx")
    assert len(xsens_files) > 0, \
        f"No Xsens files were found in inside the directory. " \
        f"Confirm your data files or path! Path: {xsens_trial_path}"
    assert len(xsens_files) == 1, \
        f"Multiple Xsens trials were found inside the directory. " \
        f"Confirm your data files or path!  Files: {xsens_trial_path}"

    xsens_file_path = xsens_files[0]

    # extract xsens general data (.xlsx)
    import pandas as pd
    pos3s_com, segments_pos3d, segments_quat, joint_angles_euler_zxy, \
        angular_velocity, imus_free_acc, imus_mag = pd.read_excel(
            xsens_file_path,
            sheet_name=["Center of Mass",                 # position of COM in 3d space
                        "Segment Position",               # positions of joints in 3d space
                        "Segment Orientation - Quat",     # segment global orientation (sensor global orient after sensor2segment calibration)
                        "Joint Angles ZXY",               # parent relative joint orientations?
                        "Segment Angular Velocity",       # segment angular velocity (gyroscope-like data in segment referential)
                        "Sensor Free Acceleration",       # sensor free acceleration (accelerometer data without gravity vector)
                        "Sensor Magnetic Field",          # sensor magnetometer data?
                        ],
            index_col=0
        ).values()

    # add dim (S, [1], 3)  +  ignore com_vel / com_accel
    pos3s_com = np.expand_dims(pos3s_com.values, axis=1)[..., [0, 1, 2]]
    n_samples = len(pos3s_com)

    # assumes a perfect sampling freq of 60hz
    timestamps = np.arange(1, n_samples + 1) * (1 / 60.)

    # 3D positions of the origin of segments referentials'
    segments_pos3d = segments_pos3d.values.reshape(n_samples, -1, 3)

    # segment orientation quaternions
    segments_quat = segments_quat.values.reshape(n_samples, -1, 4)

    # joint angles as euler (follow zxy?)
    joint_angles_euler_zxy = np.deg2rad(joint_angles_euler_zxy.values.reshape(n_samples, -1, 3))

    # sensor data (mapped to respective segments)
    imus_free_acc = imus_free_acc.values.reshape(n_samples, -1, 3)
    imus_gyr = angular_velocity.values.reshape(n_samples, -1, 3)
    imus_mag = imus_mag.values.reshape(n_samples, -1, 3)

    return dict(acc=imus_free_acc,
                gyr=imus_gyr,
                mag=imus_mag,
                segments_quat=segments_quat,
                segments_pos=segments_pos3d,
                joints_angle=joint_angles_euler_zxy,
                center_of_mass=pos3s_com,
                timestamps=timestamps,
                num_samples=len(timestamps),
                freq=60.)


def extract_xsens_analyse_data(xsens_trial_path, resample_freq=60.0, ref_angles="tpose",
                               reset_orientation=True, reset_position=True, initial_heading=0,
                               add_g_vec=True, interval2discard_s=None, plot_data=False):
    """
    Extract relevant data from xsens .xlsx files, and applies
    some processing (frequency resampling, reset position/heading,
    change reference angles, etc...).

    Args:
        xsens_trial_path(str): path to the trial directory containing
            xsens data.
        resample_freq(float): desired output signal frequency. Xsens
            Awinda uses 60Hz by default.
        ref_angles(str): referential to return segment angles. Can be
            one of "npose", "tpose".
        reset_orientation(bool): if skeleton heading should be reset
            (facing x-axis forward).
        reset_position(bool): if skeleton root joint (XY) should be
            reset to origin.
        initial_heading(float): desired heading to initialize skeleton
            in rads.
        add_g_vec(bool): add gravity vector to accelerometer data.
        interval2discard_s(tuple[float], None): interval of data to
            discard in seconds (ex. calibration/warmup section), or
            None to keep all samples.
        plot_data(bool): if the extracted data should be plotted.

    Returns:
        (dict): extracted xsens data

    """
    assert ref_angles in ["tpose", "npose"], \
        f"Invalid referential specified ({ref_angles}) " \
        f"choose one of ['tpose', 'npose']"

    xsens_raw_data = extract_xsens_analyse_raw_data(xsens_trial_path)
    imus_acc = xsens_raw_data["acc"]
    imus_gyr = xsens_raw_data["gyr"]
    imus_mag = xsens_raw_data["mag"]
    segments_quat = xsens_raw_data["segments_quat"]
    segments_pos3d = xsens_raw_data["segments_pos"]
    joint_angles_euler_zxy = xsens_raw_data["joints_angle"]
    pos3s_com = xsens_raw_data["center_of_mass"]
    timestamps = xsens_raw_data["timestamps"]
    n_samples = xsens_raw_data["num_samples"]

    # remove unwanted data indexes
    keep_idx = list(range(len(timestamps)))
    if interval2discard_s is not None:
        keep_idx = list(set(keep_idx) - set(range(round(interval2discard_s[0] * 60.0),
                                                  round(interval2discard_s[0] * 60.0))))
        segments_pos3d = segments_pos3d[keep_idx, ...]
        segments_quat = segments_quat[keep_idx, ...]
        joint_angles_euler_zxy = joint_angles_euler_zxy[keep_idx, ...]
        imus_acc = imus_acc[keep_idx, ...]
        imus_gyr = imus_gyr[keep_idx, ...]
        imus_mag = imus_mag[keep_idx, ...]
        timestamps = timestamps[keep_idx, ...]

    if resample_freq != 60.0:
        # assumes xsens data is uniformly sampled
        from hipose.utils import resample_data_frequency_factor
        sample_factor = (resample_freq / 60.)
        pos3s_com = resample_data_frequency_factor(pos3s_com, factor=sample_factor, axis=0)
        segments_pos3d = resample_data_frequency_factor(segments_pos3d, factor=sample_factor, axis=0)
        joint_angles_euler_zxy = resample_data_frequency_factor(joint_angles_euler_zxy, factor=sample_factor, axis=0)
        segments_quat = resample_data_frequency_factor(segments_quat, factor=sample_factor, method="slerp")
        imus_acc = resample_data_frequency_factor(imus_acc, factor=sample_factor, axis=0)
        imus_gyr = resample_data_frequency_factor(imus_gyr, factor=sample_factor, axis=0)
        imus_mag = resample_data_frequency_factor(imus_mag, factor=sample_factor, axis=0)
        timestamps = resample_data_frequency_factor(timestamps, factor=sample_factor, axis=0)
        n_samples = len(timestamps)

    if reset_position:
        # reset initial XY position to origin (average first 5 samples)
        pos_ref_xyz = pos3s_com[0:5, 0].mean(axis=0)
        pos3s_com, segments_pos3d = reset_skeleton_position(
                pos_args_list=[pos3s_com, segments_pos3d],
                pos_ref=pos_ref_xyz, axis2reset=(True, True, False))

    if reset_orientation:
        # reset initial heading (z-axis) to facing forward (average first 60 samples)
        # or desired heading
        # TODO: confirm if more data also needs rotation (joint_angle root?)
        ori_ref = quat_avg(segments_quat[0:60, 0])
        (segments_quat,), (pos3s_com, segments_pos3d), (imus_acc, imus_gyr, imus_mag) = \
            reset_skeleton_orientation(
                    orient_arg_list=(segments_quat,),
                    pos_args_list=(pos3s_com, segments_pos3d),
                    vec_args_list=(imus_acc, imus_gyr, imus_mag),
                    rot_ref=ori_ref, axis2reset=(False, False, True))

        init_heading_rot = convert_euler_to_quat([0, 0, -initial_heading], seq="xyz")
        for s in range(segments_quat.shape[1]):
            segments_quat[:, s, :] = quat_mult(quat_inv(init_heading_rot), segments_quat[:, s, :])

    # convert data to desired reference referential
    # for some reason the orientation data seems to come relative to T-pose
    # while the raw imu data comes relative to the N-pose
    if ref_angles == "npose":
        segments_quat = segments_quat.transpose((1, 0, 2))
        for s, s_ori in enumerate(segments_quat):
            segments_quat[s] = quat_mult(s_ori, convert_to_rel_npose_angles[s])
        segments_quat = segments_quat.transpose((1, 0, 2))
    elif ref_angles == "tpose":
        imus_acc = imus_acc.transpose((1, 0, 2))
        imus_gyr = imus_gyr.transpose((1, 0, 2))
        imus_mag = imus_mag.transpose((1, 0, 2))
        for s, (s_acc, s_gyr, s_mag) in \
                enumerate(zip(imus_acc, imus_gyr, imus_mag)):
            imus_acc[s] = vec_rotate(s_acc, convert_to_rel_npose_angles[s])
            imus_gyr[s] = vec_rotate(s_gyr, convert_to_rel_npose_angles[s])
            imus_mag[s] = vec_rotate(s_mag, convert_to_rel_npose_angles[s])
        imus_acc = imus_acc.transpose((1, 0, 2))
        imus_gyr = imus_gyr.transpose((1, 0, 2))
        imus_mag = imus_mag.transpose((1, 0, 2))
    else:
        raise NotImplementedError

    if add_g_vec:
        #  add gravity vector to accelerometer data (based on sensor orientation)
        imus_acc = add_gravity_acceleration(
                imus_acc.reshape(-1, 3).copy(),
                segments_quat.reshape(-1, 4).copy()
        ).reshape((n_samples, -1, 3))

    if plot_data:
        ori_euler = convert_quat_to_euler_continuous(segments_quat, seq="xyz")
        # plots data between 0-30 seconds
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.rcParams.update({"legend.labelspacing": 0.125, "legend.fontsize": 10})
        pt_s, pt_e, sspl = (int(resample_freq * 0), int(resample_freq * 30), 1)
        fig, ax = plt.subplots(23, 5, figsize=(30, 60))
        for s in range(23):
            # accelerometer
            ax[s, 0].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 0].plot(timestamps[pt_s:pt_e:sspl], imus_acc[pt_s:pt_e:sspl, s])
            ax[s, 0].legend(["x", "y", "z"])
            ax[s, 0].set_title(f"Acc_imu({xsens_joint_names[s]} - {s})")
            ax[s, 0].set_ylabel("m/s²")

            # gyroscope
            ax[s, 1].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 1].plot(timestamps[pt_s:pt_e:sspl], imus_gyr[pt_s:pt_e:sspl, s])
            ax[s, 1].legend(["x", "y", "z"])
            ax[s, 1].set_title(f"Gyr_imu({xsens_joint_names[s]} - {s})")
            ax[s, 1].set_ylabel("rads/s")

            # magnetometer
            mag_norm = np.linalg.norm(imus_mag[pt_s:pt_e:sspl, s], axis=-1, keepdims=True)
            ax[s, 2].set_prop_cycle(color=["firebrick", "green", "royalblue", "gray"])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], imus_mag[pt_s:pt_e:sspl, s])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], mag_norm)
            ax[s, 2].legend(["x", "y", "z", "norm"])
            ax[s, 2].set_title(f"Mag_imu({xsens_joint_names[s]} - {s})")
            ax[s, 2].set_ylabel("a.u.")

            # orientation_test
            ax[s, 3].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 3].plot(timestamps[pt_s:pt_e:sspl], np.rad2deg(ori_euler)[pt_s:pt_e:sspl, s])
            ax[s, 3].legend(["Roll", "Pitch", "Yaw"])
            ax[s, 3].set_title(f"Segments Orientation_imu({xsens_joint_names[s]} - {s})")
            ax[s, 3].set_ylabel("degrees")

            # plot power spectrum of signal
            from scipy.fft import rfft
            fmag = np.abs(rfft(np.linalg.norm(ori_euler[:, s], axis=-1), axis=0, norm="ortho"))
            f = np.linspace(0, resample_freq / 2, len(fmag))
            s_dbf = 20 * np.log10(fmag)
            ax[s, 4].set_ylabel("Power [dB]")
            ax[s, 4].set_xlabel("Freq")
            ax[s, 4].set_title(f"PowerSpectrum_imu({xsens_joint_names[s]} - {s})")
            ax[s, 4].plot(np.squeeze(f), np.squeeze(s_dbf))

        trial_name = os.path.basename(xsens_trial_path)
        fig.suptitle(f"Xsens IMUs data (trial: {trial_name})", size=40, y=0.995)
        fig.tight_layout()
        fig.show()

    return dict(acc=imus_acc,
                gyr=imus_gyr,
                mag=imus_mag,
                segments_quat=segments_quat,
                segments_pos=segments_pos3d,
                joints_angle=joint_angles_euler_zxy,
                center_of_mass=pos3s_com,
                timestamps=timestamps,
                num_samples=len(timestamps),
                freq=resample_freq)
