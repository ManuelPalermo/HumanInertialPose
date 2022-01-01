import warnings
import re
import os
import glob
import numpy as np

from hipose.rotations import convert_quat_to_euler_continuous, \
    quat_diff, convert_euler_to_quat, quat_mult, quat_avg

from hipose.imu import compute_imu_orientation, rotate_vectors, \
    remove_gravity_acceleration, imus_static_calibration

from hipose.utils import resample_data_frequency, \
    find_low_variability_sections


def get_mtmanager_trial_files(mtmanager_trial_path, extension=".csv"):
    """Find all files of an mtmanager trial inside a directory.

    Args:
        mtmanager_trial_path(str): path to the trial directory
            containing mtmanager files.

    Returns:
        (dict[str, str]): dict with mtmanager sensor serial-number as
            keys and respective data paths as values.

    """
    mtmanager_file_paths = dict()
    sensor_name_pattern = re.compile(".*_(.*).csv")
    for f in sorted(glob.glob(mtmanager_trial_path + "/*" + extension)):
        match = re.match(sensor_name_pattern, f)
        if match is not None:
            sensor_id = match.group(1).upper()
            mtmanager_file_paths[sensor_id] = f
    return mtmanager_file_paths


def _fix_mtmanager_raw_data_axis(acc, gyr, mag):
    """
    Corrects sensor axis for NWU.
    """
    acc *= [-1., -1., -1.]
    gyr *= [ 1.,  1.,  1.]
    mag *= [ 1.,  1.,  1.]
    return acc, gyr, mag


def extract_mtmanager_raw_data(mtmanager_trial_path, sensor_order=None,
                               delimiter=";", interpolate_missing=True,
                               verbose=False):
    """
    Extract available data from the mtmanager files.

    Args:
        mtmanager_trial_path(str): path to the trial directory
            containing mtmanager data.
        sensor_order (None|list[str]): serial-number of sensors to
            order into array. Or none if alphabetical load order should
            be used.
        delimiter(str): delimiter used on trial files.
        interpolate_missing(bool): if missing data packets should
            be interpolated.
        verbose(bool): if warning messages should be printed.

    Returns:
        (dict): extracted imu data

    """
    mtmanager_file_paths = get_mtmanager_trial_files(mtmanager_trial_path)
    assert len(mtmanager_file_paths) > 0, f"No mtmanager files were found in inside the " \
                                          f"directory. Confirm your data files or path! " \
                                          f"Path: {mtmanager_trial_path}"

    if sensor_order is not None:
        # reorder dict desired sensor order
        # (assumes use of python3.6+ which uses ordered dict implementation)
        mtmanager_file_paths = {k: mtmanager_file_paths[k] for k in sensor_order}

    # determine start / end index which is compatible with all files
    # (needs to read all files before hand)
    import pandas as pd
    sidx, eidx = (np.inf, np.inf)
    for snumber, spath in mtmanager_file_paths.items():
        _sdata = pd.read_csv(spath, delimiter=delimiter, header=4,
                             engine="python", index_col=0)
        _sdata.set_index(np.unwrap(_sdata.index, period=2 ** 16), inplace=True)   # unwrap 2^16
        _s_sidx, _s_eidx = (_sdata.index[0], _sdata.index[-1] + 1)
        sidx = (_s_sidx if (_s_sidx < sidx) else sidx)
        eidx = (_s_eidx if (_s_eidx < eidx) else eidx)

    # load mtmanager data from all files in trial
    metadata = []
    data = dict()
    for snumber, spath in mtmanager_file_paths.items():
        # read metadata from the start of file
        smetadata = dict()
        with open(spath) as sfile:
            metadata_match_pattern = re.compile("// (.*): (.*)")
            while True:
                fline = sfile.readline()
                match = re.match(metadata_match_pattern, fline)
                if match is not None:
                    meta_id, meta_value = match.group(1), match.group(2)
                    smetadata[meta_id] = meta_value
                else:
                    break

        # read data from file
        sdata = pd.read_csv(spath, delimiter=delimiter, header=4, engine="python", index_col=0)

        # infer which columns are contained in the file and store as metadata
        cols = list(
            set([
                c.lower().replace("_x", "").replace("_y", "").replace("_z", "")
                    .replace("_q0", "").replace("_q1", "").replace("_q2", "").replace("_q3", "")
                for c in sdata.columns
            ]))

        # remove discontinuity on idx=2^16
        sdata.set_index(np.unwrap(sdata.index, period=2 ** 16), inplace=True)

        # check for lost samples of data and interpolate missing
        n_lost_samples = (np.diff(sdata.index) - 1).sum()
        if verbose and n_lost_samples:
            warnings.warn(f"\nThe Sensor({snumber}) contains a total of ({n_lost_samples}) "
                          f"samples of data which have been lost! Trying to interpolate!")

        # interpolate nans on middle of trial
        if interpolate_missing:
            # TODO: some data cannot be interpolated columnwise (e.g quats - although might be acceptable for small changes)

            # add missing data rows
            sdata = sdata.reindex(index=np.arange(sidx, eidx),
                                  fill_value=np.nan).reset_index()

            sdata.interpolate(method="polynomial", order=2, limit=30,
                              limit_area='inside', limit_direction="both",
                              inplace=True)
            # interpolate nans on trial start/end
            sdata.interpolate(method="linear", limit=10,
                              limit_area="outside", limit_direction="both",
                              inplace=True)

        # extracted data and metadata for each sensor
        smetadata["columns"] = list(cols)
        smetadata["nsamples"] = len(sdata)
        smetadata["freq"] = float(smetadata["Update Rate"].replace("Hz", ""))
        metadata.append(smetadata)
        data[snumber.upper()] = sdata

    # check if metadata from all sensors is the same
    assert all(elem == metadata[0] for elem in metadata), \
        "Not all files contain the same metadata, " \
        "this signals a problem with the data which " \
        "should be corrected!"
    metadata = metadata[0]

    data_freq = metadata["freq"]
    nsamples = metadata["nsamples"]
    nsensors = len(mtmanager_file_paths)

    # infer timestamps from sampling frequency (assumed to be perfect)
    timestamps = np.arange(1, nsamples + 1) * 1 / data_freq

    # join data from all sensors into the same dataframe (with desired order)
    merged_data = pd.concat([d.add_prefix(snum + "_")
                             for snum, d in data.items()],
                            axis=1)

    # select data from each modality and convert to numpy array (nsamples, sensors, dim)
    mtdata = dict()
    for data_modality in metadata["columns"]:
        mdata = merged_data[
            [c for c in merged_data.columns if data_modality in c.lower()]
        ].values.reshape(nsamples, nsensors, -1)
        mtdata[data_modality] = mdata

    mtdata["acc"], mtdata["gyr"], mtdata["mag"] = _fix_mtmanager_raw_data_axis(
            **{k: v for k, v in mtdata.items() if k in ("acc", "gyr", "mag")})

    if interpolate_missing:
        # check if data is valid (truncate trajectory to last valid samples if not)
        miss_acc = np.where(np.isnan(mtdata["acc"]));  n_missing_acc = len(mtdata["acc"][miss_acc])
        miss_gyr = np.where(np.isnan(mtdata["gyr"]));  n_missing_gyr = len(mtdata["gyr"][miss_gyr])
        miss_mag = np.where(np.isnan(mtdata["mag"]));  n_missing_mag = len(mtdata["mag"][miss_mag])
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
            mtdata["acc"] = mtdata["acc"][:last_valid_idx]
            mtdata["gyr"] = mtdata["gyr"][:last_valid_idx]
            mtdata["mag"] = mtdata["mag"][:last_valid_idx]
            timestamps = timestamps[:last_valid_idx]

            if verbose:
                warnings.warn(f"\nMissing data samples which could not be "
                              f"interpolated(>30 consecutive) were found: "
                              f"\nAcc - idx:[{corrupt_s_acc} - {corrupt_e_acc}]"
                              f"   |   Gyr - idx:[{corrupt_s_gyr} - {corrupt_e_gyr}]"
                              f"   |   Mag - idx:[{corrupt_s_mag} - {corrupt_e_mag}]"
                              f"\nTruncating trajectory to last sample of valid data "
                              f"([0-{n_init_samples}] -> [0-{last_valid_idx}])!")

    assert nsamples > 50, \
        "Data is corrupted (less than 50 samples usable)!"
    assert len(timestamps) == len(mtdata["acc"]) == len(mtdata["gyr"]) == len(mtdata["mag"]), \
        "Not all extracted data has the same number of samples."

    return dict(**mtdata,
                data_modalities=list(mtdata.keys()),
                sensor_ids=list(data.keys()),
                timestamps=timestamps,
                num_samples=nsamples,
                freq=data_freq)


def extract_mtmanager_data(mtmanager_trial_path, orient_algorithm="madgwick",
                           resample_freq=60.0, calib_range_s=(0.0, 5.0), ignore_mag=True,
                           s2s_calib_method="static_mag", remove_g_vec=False,
                           interval2discard_s=None, plot_data=False,
                           sensor_order=None, mapping_sensorid_to_segname=None,
                           imus_manual_alignment=None, output_ref_data_transform=None):
    """
    Extract the following data from the mtmanager files:
    imu_acc_data, imu_gyr_data, imu_mag_data, segment_orientations
    and timestamps (relative to first sample). IMU data is converted
    to be in segment referential (not in sensor referential).

    Args:
        mtmanager_trial_path(str): path to the trial directory
            containing mtmanager data.
        orient_algorithm(None|str): algorithm used to calculate each of
            the sensors' orientation. Can be one of [None, 'Integral',
            'Mahony', 'Madgwick', 'EKF']. If None, uses the orientation
            from the sensor onboard fusion . Defaults to 'Madgwick'.
        resample_freq(float): desired output signal frequency. mtmanager
            data uses 100hz sampling.
        calib_range_s(tuple[float]): data range (in seconds) to consider
            for standing calibration (the user should be in 'N-Pose'
            during this time). If None, then no calibration is applied
            to the data.
        ignore_mag(bool): if magnetometer data should be ignored when
            computing sensor orientations.
        ref_angles(str): referential to return segment angles. Can be
            one of ["npose", "tpose"].
        s2s_calib_method(None|str): sensor to segment calibration method
            to use. Can be [None, "manual", "static", "static_mag"].
        remove_g_vec(bool): if gravity vector should be removed
            from accelerometer data.
        interval2discard_s(tuple[float], None): interval of data to
            discard in seconds (ex. calibration/warmup section). Or None
            to keep all samples.
        plot_data(bool): Plots the data extracted from each sensor.
        mapping_sensorid_to_segname(None|dict[str,str]): convert sensor
            IDs to segment names.
        sensor_order (None|list[str]): serial-number of sensors to
            order into array. Or none if alphabetical load order should
            be used.
        imus_manual_alignment(None|np.ndarray): rotations to apply to
            each sensor to manually rotate it to desired orientation,
            for s2s_calib.
        output_ref_data_transform(None|np.ndarray[Sx4]): apply a
            referential transformation to output sensor data.

    Returns:
        (dict): extracted mtmanager data

    TODO:
        - add option to filter data (band-pass butterworth) (necessary for heavy downsampling)?

    """
    assert s2s_calib_method in [None, "manual", "static", "static_mag"], \
        f"Invalid sensor-to-segment calibration method ({s2s_calib_method})," \
        f" choose one of  [None, 'manual', 'static', 'static_mag']"

    mtmanager_raw_data = extract_mtmanager_raw_data(mtmanager_trial_path,
                                                    sensor_order=sensor_order,
                                                    interpolate_missing=True)

    imu_names = ([mapping_sensorid_to_segname[sid] for sid in mtmanager_raw_data["sensor_ids"]]
                 if mapping_sensorid_to_segname is not None
                 else mtmanager_raw_data["sensor_ids"])

    acc_data = mtmanager_raw_data["acc"]
    gyr_data = mtmanager_raw_data["gyr"]
    mag_data = mtmanager_raw_data["mag"]
    timestamps = mtmanager_raw_data["timestamps"]
    data_freq = mtmanager_raw_data["freq"]
    has_imu_ori = "quat" in mtmanager_raw_data
    if has_imu_ori:
        ori_data = mtmanager_raw_data["quat"]
        ori_data /= np.linalg.norm(ori_data, axis=-1, keepdims=True)

    assert orient_algorithm is not None or has_imu_ori, \
        "Orientation algorithm cannot be None if no quaternion " \
        "orientation data is present in the raw mtmanager files!"

    # from scipy.signal import butter, filtfilt
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
        if has_imu_ori:
            ori_data = resample_data_frequency(ori_data, orig_t=timestamps,
                                               target_t=target_t, method="slerp")
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
    if has_imu_ori:
        ori_data = ori_data.transpose((1, 0, 2))

    orient_data = []
    for s, (s_acc, s_gyr, s_mag) in enumerate(zip(acc_data, gyr_data, mag_data)):
        if calib_range_s is not None and s2s_calib_method is not None:
            # apply s2s rotation to imu data and orientation
            s_acc, s_gyr, s_mag = rotate_vectors(
                    s_acc, s_gyr, s_mag,
                    rotation=calib_params["s2s_offset"][s],
                    inv=True)

        if orient_algorithm is None and has_imu_ori:
            # TODO: Seems to give expected results, but not very logical
            #  (prob bugs, check transforms!)
            #  (used for getting some results in the meantime)

            # convert from ENU (default used by mtmanager) to NWU
            s_ori_data = quat_mult(
                convert_euler_to_quat([0, 0, np.pi/2], seq="XYZ"),
                ori_data[s]
            )

            # apply s2s referential transformation to sensor
            imu_orient = (quat_mult(s_ori_data, calib_params["s2s_offset"][s])
                          if calib_range_s is not None and s2s_calib_method is not None
                          else s_ori_data)

            # reset sensor orientation to be relative to first samples
            # (is different from inertial data referential?)
            imu_orient = quat_diff(imu_orient, quat_avg(imu_orient[calib_sidx:calib_eidx]))

        else:
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
        # leading to inconsistency, so instead we just return
        # the data relative to magnetic-north when using magnetometer
        # and also initial_heading for outside functions if needed
        from hipose.imu import magnetometer_heading
        # avg first 60 samples from all sensors used for reference
        # (should all be in global frame at this stage)
        ref_sensors = [i for i in range(nimus)]
        initial_heading = magnetometer_heading(mag=np.mean(mag_data[:60, ref_sensors], axis=(0, 1)),
                                               acc=np.mean(acc_data[:60, ref_sensors], axis=(0, 1)),
                                               frame="NWU")

    if output_ref_data_transform is not None:
        # convert from N-pose relative referential to T-pose
        for s in range(nimus):
            orient_data[:, s, :] = quat_mult(orient_data[:, s, :], output_ref_data_transform[s])
            acc_data[:, s, :], gyr_data[:, s, :], mag_data[:, s, :] = rotate_vectors(
                    acc_data[:, s, :], gyr_data[:, s, :], mag_data[:, s, :],
                    rotation=output_ref_data_transform[s],
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
        plt.rcParams.update({"legend.labelspacing": 0.125, "legend.fontsize": 10})
        pt_s, pt_e, sspl = (int(resample_freq * 0), int(resample_freq * 30), 1)
        fig, ax = plt.subplots(nimus, 5, figsize=(30, 60))
        for s in range(nimus):
            # accelerometer
            ax[s, 0].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 0].plot(timestamps[pt_s:pt_e:sspl], acc_data[pt_s:pt_e:sspl, s])
            ax[s, 0].legend(["x", "y", "z"])
            ax[s, 0].set_title(f"Acc_imu({imu_names[s]} - {s})")
            ax[s, 0].set_ylabel("m/s²")
            ax[s, 0].axvline(trial_start_idx * (1 / resample_freq), ls="--", color="k")

            # gyroscope
            ax[s, 1].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 1].plot(timestamps[pt_s:pt_e:sspl], gyr_data[pt_s:pt_e:sspl, s])
            ax[s, 1].legend(["x", "y", "z"])
            ax[s, 1].set_title(f"Gyr_imu({imu_names[s]} - {s})")
            ax[s, 1].set_ylabel("rads/s")
            ax[s, 1].axvline(trial_start_idx * (1 / resample_freq), ls="--", color="k")

            # magnetometer
            mag_norm = np.linalg.norm(mag_data[pt_s:pt_e:sspl, s], axis=-1, keepdims=True)
            ax[s, 2].set_prop_cycle(color=["firebrick", "green", "royalblue", "gray"])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], mag_data[pt_s:pt_e:sspl, s])
            ax[s, 2].plot(timestamps[pt_s:pt_e:sspl], mag_norm / mag_norm.mean())
            ax[s, 2].legend(["x", "y", "z", "norm"])
            ax[s, 2].set_title(f"Mag_imu({imu_names[s]} - {s})")
            ax[s, 2].set_ylabel("a.u.")
            ax[s, 2].axvline(trial_start_idx * (1 / resample_freq), ls="--", color="k")

            # orientation_test
            ax[s, 3].set_prop_cycle(color=["firebrick", "green", "royalblue"])
            ax[s, 3].plot(timestamps[pt_s:pt_e:sspl],
                          np.rad2deg(orient_data_euler)[pt_s:pt_e:sspl, s])
            ax[s, 3].legend(["Roll", "Pitch", "Yaw"])
            ax[s, 3].set_title(f"Segments Orientation({orient_algorithm})_imu({imu_names[s]} - {s})")
            ax[s, 3].set_ylabel("degrees")
            ax[s, 3].axvline(trial_start_idx * (1 / resample_freq), ls="--", color="k")

            # plot power spectrum of signal
            from scipy.fft import rfft
            fmag = np.abs(rfft(np.linalg.norm(orient_data_euler[:, s], axis=-1),
                               axis=0, norm="ortho"))  # spectrum
            f = np.linspace(0, resample_freq / 2, len(fmag))  # frequencies
            s_dbf = 20 * np.log10(fmag)  # to decibels
            ax[s, 4].set_ylabel("Power [dB]")
            ax[s, 4].set_xlabel("Freq")
            ax[s, 4].set_title(f"PowerSpectrum_imu({imu_names[s]} - {s})")
            ax[s, 4].plot(np.squeeze(f), np.squeeze(s_dbf))

        trial_name = os.path.basename(mtmanager_trial_path)
        fig.suptitle(f"MtManager IMUs data (trial: {trial_name})", size=40, y=0.995)
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


def extract_mtwawinda_data(mtwawinda_trial_path, mapping_sensorid_to_segname,
                           orient_algorithm="madgwick", resample_freq=60.0,
                           calib_range_s=(0.0, 5.0), ignore_mag=True,
                           s2s_calib_method="static_mag", remove_g_vec=False,
                           interval2discard_s=None, plot_data=False,
                           imus_manual_alignment=None, ref_angles="npose"):

    """
    Wrapper util function for MtManager exported data, when using the
    MTwAwinda sensors.

    Args:
        mtwawinda_trial_path(str): path to the trial directory
            containing mtwawinda data.
        mapping_sensorid_to_segname(dict[str,str]): convert sensor
            IDs to segment names.
        orient_algorithm(None|str): algorithm used to calculate each of
            the sensors' orientation. Can be one of [None, 'Integral',
            'Mahony', 'Madgwick', 'EKF']. If None, uses the orientation
            from the sensor onboard fusion . Defaults to 'Madgwick'.
        resample_freq(float): desired output signal frequency. mtmanager
            data uses 100hz sampling.
        calib_range_s(tuple[float]): data range (in seconds) to consider
            for standing calibration (the user should be in 'N-Pose'
            during this time). If None, then no calibration is applied
            to the data.
        ignore_mag(bool): if magnetometer data should be ignored when
            computing sensor orientations.
        ref_angles(str): referential to return segment angles. Can be
            one of ["npose", "tpose"].
        s2s_calib_method(None|str): sensor to segment calibration method
            to use. Can be [None, "manual", "static", "static_mag"].
        remove_g_vec(bool): if gravity vector should be removed
            from accelerometer data.
        interval2discard_s(tuple[float], None): interval of data to
            discard in seconds (ex. calibration/warmup section). Or None
            to keep all samples.
        plot_data(bool): Plots the data extracted from each sensor.
        sensor_order (None|list[str]): serial-number of sensors to
            order into array. Or none if alphabetical load order should
            be used.
        imus_manual_alignment(None|np.ndarray): rotations to apply to
            each sensor to manually rotate it to desired orientation,
            for s2s_calib. Can also be "default" to use some
            approximate default rotation when sensor placement follows
            xsens guidelines.
        ref_angles(str): referential to return segment angles. Can be
            one of "npose" or "tpose".

    Returns:
        (dict): extracted mtwawinda data.

    """
    # default sensor order to follow xsens analyse
    mtwawinda_segment_order = (
            "pelvis", "stern", "head",
            "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",
            "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",
            "right_upper_leg", "right_lower_leg", "right_foot",
            "left_upper_leg", "left_lower_leg", "left_foot",
    )

    # checks if correct mapping from sensor IDs to segment names is provided
    assert ((isinstance(mapping_sensorid_to_segname, dict))
            and (sorted(list(mapping_sensorid_to_segname.values()))
                 == sorted(list(mtwawinda_segment_order))))

    if isinstance(imus_manual_alignment, str) and imus_manual_alignment == "default":
        # default rotations which align IMUs data from local referential (based on placement)
        # to world referential (NWU frame: x-axis forward / y-axis left / z-axis up)
        # (assumes placement is the same - see xsens recommended placement,
        # sensor led should be on the bottom side when placed)
        # (ignores orientation offsets)
        imus_manual_alignment = \
            convert_euler_to_quat(
                    # torso/head IMUs
                    [[0,           np.pi / 2,  0        ],
                     [0,           np.pi / 2,  np.pi    ],
                     [0,           np.pi / 2,  0        ],
                     # Right Arm IMUs
                     [np.pi / 2,   0,         -np.pi / 2],
                     [0,           np.pi / 2, -np.pi / 2],
                     [0,           np.pi / 2, -np.pi / 2],
                     [0,           np.pi / 2, -np.pi / 2],
                     # Left Arm IMUs
                     [np.pi / 2,   0,         -np.pi / 2],
                     [0,           np.pi / 2,  np.pi / 2],
                     [0,           np.pi / 2,  np.pi / 2],
                     [0,           np.pi / 2,  np.pi / 2],
                     # Right leg IMUs
                     [0,           np.pi / 2, -np.pi / 2],
                     [np.pi / 4,   np.pi / 2,  np.pi / 2],
                     [0,          -np.pi / 6,  np.pi    ],
                     # Left leg IMUs
                     [0,           np.pi / 2,  np.pi / 2],
                     [-np.pi / 4,  np.pi / 2, -np.pi / 2],
                     [0,          -np.pi / 6,  np.pi    ],
                     ],
                    seq="XYZ")

    # assumes that calib data will be in NPose
    # need to change later to perform transform in both directions
    # or perform no change (if its already on desired reference)
    assert ref_angles in ["npose", "tpose"]

    if ref_angles == "npose":
        output_ref_data_transform = None

    elif ref_angles == "tpose":
        # from default n-pose to t-pose relative
        output_ref_data_transform = \
            convert_euler_to_quat(
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     # Right Arm IMUs
                     [0, 0, 0],
                     [np.pi / 2, 0, 0],
                     [np.pi / 2, 0, 0],
                     [np.pi / 2, 0, 0],
                     # Left Arm IMUs
                     [0, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     # Left Leg IMUs
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     # Left Leg IMUs
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     ],
                    seq="XYZ")

    segmentName_to_sensorID = {v: k for k, v in mapping_sensorid_to_segname.items()}
    mtwawinda_sensor_order = [segmentName_to_sensorID[sname]
                            for sname in mtwawinda_segment_order]

    return extract_mtmanager_data(mtmanager_trial_path=mtwawinda_trial_path,
                                  orient_algorithm=orient_algorithm,
                                  resample_freq=resample_freq, calib_range_s=calib_range_s,
                                  ignore_mag=ignore_mag, s2s_calib_method=s2s_calib_method,
                                  remove_g_vec=remove_g_vec, interval2discard_s=interval2discard_s,
                                  plot_data=plot_data,
                                  sensor_order=mtwawinda_sensor_order,
                                  mapping_sensorid_to_segname=mapping_sensorid_to_segname,
                                  imus_manual_alignment=imus_manual_alignment,
                                  output_ref_data_transform=output_ref_data_transform)


def report_mtmanager_trial_status(mtmanager_trial_path, verbose=True):
    """
    Inspects the loaded data and reports known status indicators.

    Reports:

    | - file naming scheme
    | - number of samples received
    | - amount of lost packages across all sensors
    | - data sampling frequency
    | - matching xsens files in directory
    | - exported data modalities
    | - sensor serial numbers

    Args:
        mtmanager_trial_path(str): path to a directory containing
            mtmanager trial data.
        verbose(bool): if problems should be printed to terminal.

    """
    mtmanager_files = get_mtmanager_trial_files(mtmanager_trial_path)
    assert len(mtmanager_files) > 0, f"No MtManager files were found in inside the " \
                                     f"directory. Confirm your data files or path! " \
                                     f"Path: {mtmanager_trial_path}"

    data = extract_mtmanager_raw_data(mtmanager_trial_path,
                                      sensor_order=None,
                                      interpolate_missing=True)

    data_acc = data["acc"]  # assumes at least acc data was exported
    data_freq = data["freq"]
    sensor_ids = data["sensor_ids"]
    modalities = data["data_modalities"]

    # create report object assuming everything is correct
    report = dict(trial_path=os.path.basename(mtmanager_trial_path),
                  num_samples=data_acc.shape[0],
                  path_name="correct",
                  lost_samples=0.0,
                  sampl_freq=data_freq,
                  match_xsens_files="",
                  data_modalities=modalities,
                  sensor_ids=sensor_ids)

    # confirm file naming
    import re
    subj = "subject[0-9]{2}"
    seq = "(task|circuit|sequence|random|calibration)"
    rep = "(rep[0-9]|npose|tpose|frontpose|sideways|xsens)"
    p = re.compile(f".*{subj}/{subj}_{seq}/{subj}_{seq}_{rep}.*")
    trial_match = re.match(p, mtmanager_trial_path)
    if trial_match is None:
        report["path_name"] = "incorrect"

    n_lost_samples = np.isnan(data_acc).sum()
    if n_lost_samples:
        report["lost_samples"] = round(n_lost_samples / (data_acc.shape[0]*data_acc.shape[1]), 4)

    # check matching xsens files
    if mtmanager_trial_path[-1] == "/":
        # small fix for extra "/" bug
        mtmanager_trial_path = mtmanager_trial_path[:-1]

    match_xsens_files = [ext for ext in ("mvn", "mvnx", "xlsx", "c3d")
                         if os.path.isfile(f"{mtmanager_trial_path}/"
                                           f"{os.path.basename(mtmanager_trial_path)}.{ext}")]
    report["match_xsens_files"] = ", ".join(match_xsens_files)

    if verbose:
        # print reports
        print("\nChecking data from trial: ", report['trial_path'])
        print(f" - The trial contains ({report['num_samples']}) data samples!")

        if report["path_name"] == "incorrect":
            print(" - File naming does not comply with the database naming scheme!")

        if n_lost_samples:
            print(f" - Trial contains ({n_lost_samples}/{(data_acc.shape[0]*data_acc.shape[1])})"
                  f" packages which have been lost!")

        if not np.allclose(60.0, report["sampl_freq"], atol=0.2):
            print(f" - Sampling frequency mismatch: sampl_freq = {round(report['sampl_freq'], 3)}Hz!")

        if not match_xsens_files:
            print(" - No matching xsens file exist in the directory (.mvn, .mvnx, .xlsx, .c3d)!")

    return report
