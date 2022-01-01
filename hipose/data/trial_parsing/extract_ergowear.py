
import warnings
import re
import os
import glob
import numpy as np

from hipose.rotations import convert_quat_to_euler_continuous, \
    convert_euler_to_quat, quat_mult

from hipose.imu import compute_imu_orientation, rotate_vectors, \
    remove_gravity_acceleration, imus_static_calibration, apply_imu_calibration

from hipose.utils import resample_data_frequency, \
    find_low_variability_sections, select_idx_dict_of_arrays


# default rotations which align IMUs data from local referential (based on placement)
# to world referential (NWU frame: x-axis forward / y-axis left / z-axis up)
# (assumes placement is the same and ignores orientation offsets)
default_ergowear_imus_manual_alignment = \
    convert_euler_to_quat(
            # Back IMUs
            [[-np.pi / 2, 0, np.pi / 2],
             [-np.pi / 2, 0, np.pi / 2],
             [-np.pi / 2, 0, np.pi / 2],
             # Right Arm IMUs
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0],
             # Left Arm IMUs
             [-np.pi / 2, 0, np.pi],
             [-np.pi / 2, 0, np.pi],
             [-np.pi / 2, 0, np.pi]],
            seq="XYZ")

# rotations to convert ergowear imu orientations,
# from default n-pose to t-pose relative
convert_to_rel_tpose_angles = \
    convert_euler_to_quat(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             # Right Arm IMUs
             [np.pi / 2, 0, 0],
             [np.pi / 2, 0, 0],
             [np.pi / 2, 0, 0],
             # Left Arm IMUs
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0],
             [-np.pi / 2, 0, 0], ],
            seq="XYZ")


def _apply_factory_calibration(acc_data, gyr_data, mag_data):
    """Apply factory calibration to sensors offline. Was only
    computed after data had been acquired.

    """
    # acc_data = acc_data   # not being applied
    # gyr_data = gyr_data   # not being applied

    # offline hard-iron calibration (computed by rotating sensor 360º over all axis)
    mag_hard_iron_calib = np.array(
            [[ 4.0062, 21.93675,  -15.75590],
             [10.4723,  6.63245,  -32.30247],
             [19.8107, 27.14950,  -48.31928],
             [24.2664, 15.32335,  -10.71130],
             [18.0419, 33.62995,  -18.57780],
             [ 9.6640, 13.97360,   -3.14225],
             [24.0945, 37.34183,  -13.39435],
             [ 9.5920, 15.46360,  -21.30920],
             [ 2.6532,  9.93810,  -15.85080]]
    )

    mag_data = mag_data - mag_hard_iron_calib
    return acc_data, gyr_data, mag_data


def _fix_ergowear_raw_data_axis(acc_data, gyr_data, mag_data):
    """
    Correct sensor axis for NWU.
    """
    acc_data *= [-1.,  1.,  1.]
    gyr_data *= [ 1., -1., -1.]
    mag_data *= [ 1., -1., -1.]
    return acc_data, gyr_data, mag_data


def get_ergowear_trial_files(ergowear_trial_path):
    """Find all files of an ergowear trial inside a directory.

    Args:
        ergowear_trial_path(str): path to the trial directory
            containing ergowear files.

    Returns:
        (list[str]): list with all ergowear trial paths sorted.

    """
    ergowear_file_paths = [p[0] for p in
                           sorted([(f, int(re.findall(r'.*/(\d+).txt', f)[0]))
                                   for f in glob.glob(ergowear_trial_path + "/*.txt")],
                                  key=lambda x: x[1])]
    return ergowear_file_paths


def extract_ergowear_raw_data(ergowear_trial_path, ignore_mag=True,
                              interpolate_missing=True, verbose=False):
    """
    Extract raw IMU data from the ergowear dataset (acc, gyr, mag).

    Args:
        ergowear_trial_path(str): path to the trial directory
            containing ergowear data.
        ignore_mag(bool): if magnetometer data should be ignored when
            checking errors.
        interpolate_missing(bool): if missing data packets should
            be interpolated.
        verbose(bool): if warning messages should be printed.

    Returns:
        (dict): extracted raw imu data

    """
    ergowear_file_paths = get_ergowear_trial_files(ergowear_trial_path)
    assert len(ergowear_file_paths) > 0, f"No Ergowear files were found in inside the " \
                                         f"directory. Confirm your data files or path! " \
                                         f"Path: {ergowear_trial_path}"

    # load ergowear data from all files in trial
    import pandas as pd
    data = pd.concat([pd.read_csv(f, delimiter=" |,", header=None,
                                  engine="python", index_col=0)
                      for f in sorted(ergowear_file_paths)],
                     ignore_index=False)

    # set names to columns
    cols = [f"imu{s}_{mod}_{axis}"
            for s in range(1, 10)
            for mod in ["acc", "gyr", "mag"]
            for axis in ["x", "y", "z"]
            ] + ["timestamp"]

    # if data contains temperature columns ignore them (added to newer files)
    if len(data.columns) > 90:
        for idx in range(1, 10):
            temp_idx = (10 - idx) * 9
            cols.insert(temp_idx, f"imu{idx}_temp")

    data.columns = cols

    # check for lost samples of data
    n_lost_samples = (np.diff(data.index) - 1).sum()
    if verbose and n_lost_samples:
        warnings.warn(f"\nThe trial contains a total of ({n_lost_samples}) "
                      f"samples of data which have been lost! Trying to interpolate!")

    if interpolate_missing:
        # interpolate missing data values for each column
        # - interpolates lost row samples + randomly missing samples
        # - uses cubic spline up until 30 consecutive missing values
        # interpolate nans on middle of trial
        data[data == 0] = np.nan  # signal 0 values as nans
        data = data.reindex(index=np.arange(data.index[0], data.index[-1] + 1),
                            fill_value=np.nan).reset_index()    # add missing data rows

        data.interpolate(method="polynomial", order=2, limit=30,
                         limit_area='inside', limit_direction="both",
                         inplace=True)
        # interpolate nans on trial start
        data.interpolate(method="linear", limit=10,
                         limit_area="outside", inplace=True,
                         limit_direction="backward")

    # parse sensor data modalities
    acc_data = data[[c for c in cols if "acc" in c]].values.reshape(len(data), -1, 3)  # acc data in m/s²
    gyr_data = data[[c for c in cols if "gyr" in c]].values.reshape(len(data), -1, 3)  # gyr data in rads/s
    mag_data = data[[c for c in cols if "mag" in c]].values.reshape(len(data), -1, 3)  # mag data in uT?

    # apply factory calibration
    acc_data, gyr_data, mag_data = _apply_factory_calibration(acc_data, gyr_data, mag_data)

    # fix raw data referentials (appear to be incorrect for some reason)
    acc_data, gyr_data, mag_data = _fix_ergowear_raw_data_axis(acc_data, gyr_data, mag_data)

    # convert timestamps from ns to s
    timestamps = data["timestamp"].values * 1e-9

    # determine real data acquisition freq from timestamps
    data_freq = (np.diff(np.arange(len(timestamps))) / np.diff(timestamps)).mean()

    # check if data is valid (truncate trajectory to last valid samples if not)
    miss_acc = np.where(np.isnan(acc_data));  n_missing_acc = len(acc_data[miss_acc])
    miss_gyr = np.where(np.isnan(gyr_data));  n_missing_gyr = len(gyr_data[miss_gyr])
    miss_mag = np.where(np.isnan(mag_data));  n_missing_mag = len(mag_data[miss_mag])

    if ignore_mag:
        # ignore leftover nans in mag data and convert them to 0 so data it be visualized
        n_missing_mag = 0
        mag_data = np.nan_to_num(mag_data)

    if not (n_missing_acc == n_missing_gyr == n_missing_mag == 0):
        n_init_samples = len(timestamps)

        last_valid_idx_acc, corrupt_s_acc, corrupt_e_acc = \
            (((miss_acc[0][0] - 1), miss_acc[0][0], miss_acc[0][-1])
             if n_missing_acc > 0 else (n_init_samples, None, None))

        last_valid_idx_gyr, corrupt_s_gyr, corrupt_e_gyr = \
            (((miss_gyr[0][0] - 1), miss_gyr[0][0], miss_gyr[0][-1])
                if n_missing_gyr > 0 else (n_init_samples, None, None))

        last_valid_idx_mag, corrupt_s_mag, corrupt_e_mag = \
            (((miss_mag[0][0] - 1), miss_mag[0][0], miss_mag[0][-1])
             if n_missing_mag > 0 else (n_init_samples, None, None))

        last_valid_idx = max(0, min(last_valid_idx_acc, last_valid_idx_gyr, last_valid_idx_mag))
        acc_data = acc_data[:last_valid_idx]
        gyr_data = gyr_data[:last_valid_idx]
        mag_data = mag_data[:last_valid_idx]
        timestamps = timestamps[:last_valid_idx]

        if verbose:
            warnings.warn(f"\nMissing data samples which could not be "
                          f"interpolated(>30 consecutive) were found: "
                          f"\nAcc - idx:[{corrupt_s_acc} - {corrupt_e_acc}]"
                          f"   |   Gyr - idx:[{corrupt_s_gyr} - {corrupt_e_gyr}]"
                          f"   |   Mag - idx:[{corrupt_s_mag} - {corrupt_e_mag}]"
                          f"\nTruncating trajectory to last sample of valid data "
                          f"([0-{n_init_samples}] -> [0-{last_valid_idx}])!")

    assert len(timestamps) > 50, \
        'Data is corrupted (less than 50 samples usable)!'
    assert len(timestamps) == len(acc_data) == len(gyr_data) == len(mag_data), \
        "Not all extracted data has the same number of samples."

    return dict(acc=acc_data,
                gyr=gyr_data,
                mag=mag_data,
                timestamps=timestamps,
                num_samples=len(timestamps),
                freq=data_freq)


def extract_ergowear_data(ergowear_trial_path, orient_algorithm="madgwick",
                          resample_freq=100.0, calib_range_s=(0.0, 5.0),
                          ignore_mag=True, ref_angles="npose", s2s_calib_method="static",
                          imus_manual_alignment=default_ergowear_imus_manual_alignment,
                          remove_g_vec=False, interval2discard_s=None, plot_data=False):
    """
    Extract the following data from the Ergowear files:
    imu_acc_data, imu_gyr_data, imu_mag_data, segment_orientations
    and timestamps (relative to first sample). IMU data is converted
    to be in segment referential (not in sensor referential).

    Args:
        ergowear_trial_path(str): path to the trial directory
            containing ergowear data.
        orient_algorithm(str): algorithm used to calculate each of the
            sensors' orientation. Can be one of ['Integral', 'Mahony',
            'Madgwick', 'EKF']. Defaults to 'Madgwick'.
        resample_freq(float): desired output signal frequency. Ergowear
            data uses 100hz sampling.
        calib_range_s(tuple[float]): data range (in seconds) to consider
            for standing calibration (the user should be in 'N-Pose'
            during this time). If None, then no calibration is applied
            to the data.
        ignore_mag(bool): if magnetometer data should be ignored when
            computing sensor orientations.
        ref_angles(str): referential to return segment angles. Can be
            one of "npose" or "tpose".
        s2s_calib_method(None|str): static sensor to segment calibration
            method to use. Can be one of [None, "manual", "static",
            "static_mag"].
        imus_manual_alignment(None|np.ndarray): rotations to apply to
            each sensor to manually rotate it to desired orientation,
            for s2s_calib.
        remove_g_vec(bool): if gravity vector should be removed
            from accelerometer data.
        interval2discard_s(tuple[float], None): interval of data to
            discard in seconds (ex. calibration/warmup section). Or None
            to keep all samples.
        plot_data(bool): Plots the data extracted from each sensor.

    Returns:
        (dict): extracted ergowear data

    TODO:
        - add option to filter data (band-pass butterworth) (necessary for heavy downsampling)?

    """
    assert ref_angles in ["tpose", "npose"], \
        f"Invalid referential specified ({ref_angles}) " \
        f"choose one of ['tpose', 'npose']"

    assert s2s_calib_method in [None, "manual", "static", "static_mag"], \
        f"Invalid sensor-to-segment calibration method ({s2s_calib_method})," \
        f" choose one of  [None, 'manual', 'static', 'static_mag']"

    imu_names = ["s1", "t4", "head", "rua", "rla", "rh", "lua", "lla", "lh"]
    ergowear_raw_data = extract_ergowear_raw_data(ergowear_trial_path,
                                                   ignore_mag=ignore_mag,
                                                   interpolate_missing=True)
    acc_data = ergowear_raw_data["acc"]
    gyr_data = ergowear_raw_data["gyr"]
    mag_data = ergowear_raw_data["mag"]
    timestamps = ergowear_raw_data["timestamps"]
    data_freq = ergowear_raw_data["freq"]

    #from scipy.signal import butter, filtfilt
    # - add option to filter data (band-pass butterworth)

    # resample data to desired frequency
    # could be done after calculating orientation (higher freq might improve filter results)
    # on the other side its more realistic to undersample raw data in final application
    if resample_freq != data_freq:
        target_t = np.linspace(0, timestamps[-1],
                               round(len(timestamps) * (resample_freq / data_freq)),
                               endpoint=True)
        acc_data = resample_data_frequency(acc_data, orig_t=timestamps, target_t=target_t)
        gyr_data = resample_data_frequency(gyr_data, orig_t=timestamps, target_t=target_t)
        mag_data = resample_data_frequency(mag_data, orig_t=timestamps, target_t=target_t)
        timestamps = resample_data_frequency(timestamps, orig_t=timestamps, target_t=target_t)

    if interval2discard_s is not None:
        keep_idx = list(set(range(len(timestamps))) -
                        set(range(round(interval2discard_s[0] * resample_freq),
                                  round(interval2discard_s[0] * resample_freq))))
        acc_data = acc_data[keep_idx, ...]
        gyr_data = gyr_data[keep_idx, ...]
        mag_data = mag_data[keep_idx, ...]
        timestamps = timestamps[keep_idx, ...]

    # metadata
    nsamples = len(timestamps)
    nimus = len(imu_names)

    calib_params = dict()
    if calib_range_s is not None:
        # compute standing calibration params of each sensor data
        # -gyro bias
        # -accelerometer magnitude
        # -magnetometer magnitude
        # -sensor to segment offsets
        calib_sidx, calib_eidx = (int(calib_range_s[0] * resample_freq),
                                  int(calib_range_s[1] * resample_freq))
        calib_params = imus_static_calibration(
                acc_data[calib_sidx:calib_eidx],
                gyr_data[calib_sidx:calib_eidx],
                mag_data[calib_sidx:calib_eidx],
                manual_align_rots=imus_manual_alignment,
                s2s_calib_method=s2s_calib_method
        )

    # compute sensor orientations using a fusion filter
    acc_data = acc_data.transpose((1, 0, 2))
    gyr_data = gyr_data.transpose((1, 0, 2))
    mag_data = mag_data.transpose((1, 0, 2))
    orient_data = []
    for s, (s_acc, s_gyr, s_mag) in enumerate(zip(acc_data, gyr_data, mag_data)):
        if calib_range_s is not None:
            # apply calibration to imu data
            s_acc, s_gyr, s_mag = apply_imu_calibration(
                    s_acc, s_gyr, s_mag,
                    **select_idx_dict_of_arrays(calib_params, axis_idx_dict={0:[s]}))

        # calculate IMUs orientation using fusion filter
        fusion_raw_imu_data = (s_acc, s_gyr) if ignore_mag else (s_acc, s_gyr, s_mag)
        imu_orient = compute_imu_orientation(*fusion_raw_imu_data,
                                             algorithm=orient_algorithm,
                                             freq=resample_freq,
                                             n_init_samples=60)
        acc_data[s] = s_acc
        gyr_data[s] = s_gyr
        mag_data[s] = s_mag
        orient_data.append(imu_orient)

    acc_data = acc_data.transpose((1, 0, 2))
    gyr_data = gyr_data.transpose((1, 0, 2))
    mag_data = mag_data.transpose((1, 0, 2))
    orient_data = np.array(orient_data).transpose((1, 0, 2))

    if remove_g_vec:
        # remove gravity vector based on sensor orientation
        acc_data = remove_gravity_acceleration(acc=acc_data.reshape(-1, 3).copy(),
                                               orient=orient_data.reshape(-1, 4).copy(),
                                               ).reshape((nsamples, nimus, 3))

    initial_heading = 0
    if not ignore_mag:
        # heading reset could be applied to segment orientation data,
        # however it does not apply correctly to magnetometer readings
        # leading to inconsistency, so instead we just return the
        # the data relative to magnetic-north when using magnetometer
        # and also initial_heading for outside functions if needed
        from hipose.imu import magnetometer_heading
        # avg first 60 samples from all sensors used for reference
        # (all should be in global frame at this stage)
        ref_sensors = [i for i in range(nimus)]
        initial_heading = magnetometer_heading(np.mean(mag_data[:60, ref_sensors], axis=(0, 1)),
                                               np.mean(acc_data[:60, ref_sensors], axis=(0, 1)))

    if ref_angles == "tpose":
        # convert from N-pose relative referential to T-pose
        for s in range(nimus):
            orient_data[:, s, :] = quat_mult(orient_data[:, s, :], convert_to_rel_tpose_angles[s])
            acc_data[:, s, :], gyr_data[:, s, :], mag_data[:, s, :] = rotate_vectors(
                    acc_data[:, s, :], gyr_data[:, s, :], mag_data[:, s, :],
                    rotation=convert_to_rel_tpose_angles[s],
                    inv=True)

    # determine idx where trial starts (after calibration) based on movement variability
    orient_data_euler = convert_quat_to_euler_continuous(orient_data, seq='xyz')
    low_var_idx = find_low_variability_sections(orient_data_euler,
                                                threshold=0.025, window_size=9,
                                                thresh_method="max")
    try:
        trial_start_idx = np.where(np.diff(low_var_idx) > 1)[0][0]
        trial_start_idx = max(0, round(trial_start_idx - (resample_freq * 0.5)))  # start - 0.5s
    except IndexError:
        trial_start_idx = 0  # no high variability idx found (or full high variability)

    if plot_data:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        # plots data between 0-30 seconds
        pt_s, pt_e, sspl = (int(resample_freq * 0), int(resample_freq * 30), 1)
        fig, ax = plt.subplots(nimus, 5, figsize=(40, 30))
        for s in range(nimus):
            # accelerometer
            ax[s, 0].set_prop_cycle(color=["firebrick", "green","royalblue"])
            ax[s, 0].plot(timestamps[pt_s:pt_e:sspl], acc_data[pt_s:pt_e:sspl, s])
            ax[s, 0].legend(["x", "y", "z"])
            ax[s, 0].set_title(f"Acc_imu({imu_names[s]} - {s})")
            ax[s, 0].set_ylabel("m/s²")
            ax[s, 0].axvline(trial_start_idx*(1/resample_freq), ls="--", color="k")

            # gyroscope
            ax[s, 1].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 1].plot(timestamps[pt_s:pt_e:sspl], gyr_data[pt_s:pt_e:sspl, s])
            ax[s, 1].legend(["x", "y", "z"])
            ax[s, 1].set_title(f"Gyr_imu({imu_names[s]} - {s})")
            ax[s, 1].set_ylabel("rads/s")
            ax[s, 1].axvline(trial_start_idx*(1/resample_freq), ls="--", color="k")

            # magnetometer
            mag_norm = np.linalg.norm(mag_data[pt_s:pt_e:sspl, s], axis=-1, keepdims=True)
            ax[s, 2].set_prop_cycle(color=["firebrick", "green", "royalblue", "gray"])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], mag_data[pt_s:pt_e:sspl, s])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], mag_norm / mag_norm.mean())
            ax[s, 2].legend(["x", "y", "z", "norm"])
            ax[s, 2].set_title(f"Mag_imu({imu_names[s]} - {s})")
            ax[s, 2].set_ylabel("a.u.")
            ax[s, 2].axvline(trial_start_idx*(1/resample_freq), ls="--", color="k")

            # orientation_test
            ax[s, 3].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 3].plot(timestamps[pt_s:pt_e:sspl],
                          np.rad2deg(orient_data_euler)[pt_s:pt_e:sspl, s])
            ax[s, 3].legend(["Roll", "Pitch", "Yaw"])
            ax[s, 3].set_title(f"Segments Orientation({orient_algorithm})_imu({imu_names[s]} - {s})")
            ax[s, 3].set_ylabel("degrees")
            ax[s, 3].axvline(trial_start_idx*(1/resample_freq), ls="--", color="k")

            # plot power spectrum of signal
            from scipy.fft import rfft
            fmag = np.abs(rfft(np.linalg.norm(orient_data_euler[:, s], axis=-1),
                               axis=0, norm="ortho"))                          # spectrum
            f = np.linspace(0, resample_freq / 2, len(fmag))                   # frequencies
            s_dbf = 20 * np.log10(fmag)                                        # to decibels
            ax[s, 4].set_ylabel("Power [dB]")
            ax[s, 4].set_xlabel("Freq")
            ax[s, 4].set_title(f"PowerSpectrum_imu({imu_names[s]} - {s})")
            ax[s, 4].plot(np.squeeze(f), np.squeeze(s_dbf))

        trial_name = os.path.basename(ergowear_trial_path)
        fig.suptitle(f"Ergoware IMUs data (trial: {trial_name})", size=40, y=0.99)
        fig.tight_layout()
        fig.show()

    return dict(acc=acc_data,
                gyr=gyr_data,
                mag=mag_data,
                segments_quat=orient_data,
                trial_start_idx=trial_start_idx,
                initial_heading=initial_heading,
                timestamps=timestamps,
                num_samples=nsamples,
                freq=resample_freq)


def report_ergowear_trial_status(ergowear_trial_path, verbose=True):
    """
    Inspects the loaded data and reports known status indicators.

    Reports:

    | - file naming scheme
    | - number of samples received
    | - start and end sample idx
    | - timestamps format and start value
    | - amount of lost packages
    | - amount of randomly missing data points
    | - data sampling frequency
    | - matching xsens files in directory

    Args:
        ergowear_trial_path(str): path to a directory containing
            ergowear trial data.
        verbose(bool): if problems should be printed to terminal.

    """
    ergowear_files = get_ergowear_trial_files(ergowear_trial_path)
    assert len(ergowear_files) > 0, f"No Ergowear files were found in inside the " \
                                    f"directory. Confirm your data files or path! " \
                                    f"Path: {ergowear_trial_path}"

    # load ergowear data from all files in trial
    import pandas as pd
    data = pd.concat([pd.read_csv(f, delimiter=(' |,'), header=None, engine="python")
                      for f in sorted(ergowear_files)],
                     ignore_index=True)
    data_indexes = data.iloc[:, 0].values
    data_imus    = data.iloc[:, 1:82].values
    data_tstamps = data.iloc[:, 82].values

    # create report object assuming everything is correct
    report = dict(trial_path=os.path.basename(ergowear_trial_path),
                  num_samples=len(data_indexes),
                  path_name="correct",
                  start_end_idx=np.nan,
                  start_tstamp_ms=0.0,
                  lost_samples=0.0,
                  random_missing=0.0,
                  sampl_freq=100.0,
                  match_xsens_files="")

    # confirm file naming
    import re
    subj = "subject[0-9]{2}"
    seq = "(task|circuit|sequence|random|validation|calibration)"
    rep = "(rep[0-9]|t[0-9]{2}|xsens)"
    p = re.compile(f".*{subj}/{subj}_{seq}/{subj}_{seq}_{rep}.*")
    trial_match = re.match(p, ergowear_trial_path)
    if trial_match is None:
        report["path_name"] = "incorrect"

    # check if there is garbage at the start of the trial
    idx_diff = np.diff(data_indexes)

    # check if indexes start in 1
    report["start_end_idx"] = f"{data_indexes[0]}  -  {data_indexes[-1]}"

    # most are around 10e9 ns
    report["start_tstamp_ms"] = round(data_tstamps[0] / 1e6, 2)

    # check number of skipped samples
    n_lost_samples = (idx_diff - 1).sum()
    if n_lost_samples:
        report["lost_samples"] = round(n_lost_samples / (len(data_indexes) + n_lost_samples), 4)

    # check number of randomly missing samples
    rnd_missing = (data_imus == 0).sum()
    if rnd_missing:
        report["random_missing"] = round(rnd_missing / np.size(data_imus), 4)

    # check sampling freq
    timestamps = data_tstamps * 1e-9
    sampl_freq = (np.diff(np.arange(len(timestamps))) / np.diff(timestamps)).mean()
    report["sampl_freq"] = round(sampl_freq, 3)

    # check matching xsens files
    if ergowear_trial_path[-1] == "/":
        # small fix for extra "/" bug
        ergowear_trial_path = ergowear_trial_path[:-1]

    match_xsens_files = [ext for ext in ("mvn", "mvnx", "xlsx", "c3d")
                         if os.path.isfile(f"{ergowear_trial_path}/"
                                           f"{os.path.basename(ergowear_trial_path)}.{ext}")]
    report["match_xsens_files"] = ", ".join(match_xsens_files)

    if verbose:
        # print reports
        print("\nChecking data from trial: ", report['trial_path'])
        print(f" - The trial contains ({report['num_samples']}) data samples!")

        if report["path_name"] == "incorrect":
            print(" - File naming does not comply with the database naming scheme!")

        if data_indexes[0] > 1:
            print(f" - Indexes are starting on ({data_indexes[0]}) instead of 1!")

        if report["start_tstamp_ms"] > 15:    # most should be around 10ms
            print(f" - Timestmps starting on ({report['start_tstamp_ms']}ms) instead of ~10ms")

        if data_tstamps.dtype not in (np.int32, np.int64, np.uint32, np.uint64, int):
            print(" - Timestamps are being saved in an incorrect format! ")

        if n_lost_samples:
            print(f" - Trial contains ({n_lost_samples}/{len(data_indexes) + n_lost_samples})"
                  f" packages which have been lost!")

        if (rnd_missing / np.size(data_imus)) > 0.005:
            print(f" - Trial contains ({rnd_missing}/{np.size(data_imus)}) randomly missing "
                  f" data points! (high values might indicate sensor connection lost)!")

        if not np.allclose(100.0, sampl_freq, atol=0.2):
            print(f" - Sampling frequency mismatch: sampl_freq = {round(sampl_freq, 3)}Hz!")

        if not match_xsens_files:
            print(" - No matching xsens file exist in the directory (.mvn, .mvnx, .xlsx, .c3d)!")

    return report
