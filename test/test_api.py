
from hipose.utils import find_resource_path


def test_fusion_filter():
    from hipose.data.trial_parsing.extract_xsens_mtmanager import extract_mtmanager_raw_data
    dataset_path = find_resource_path("./resources/sample_trials/mtwawinda/example_calibration_xsens")
    imu_data = extract_mtmanager_raw_data(dataset_path, interpolate_missing=False)

    # initialize filter fusion (example trial has 17 IMUs)
    from hipose.api.fusion_filter import InertialPoseFusionFilter
    ffilts = InertialPoseFusionFilter(
            num_imus=17,
            ignore_mag=False,
            fusion_filter_alg="madgwick",
            s2s_calib_method="static_mag",
            default_data_freq=imu_data["freq"],
    )

    # initialize calibration params from static NPose
    # (example trial has 5s of NPose at the start)
    calib_s = int(imu_data["freq"] * 5)
    ffilts.compute_imus_calibration(acc_calib_data=imu_data["acc"][0:calib_s],
                                    gyr_calib_data=imu_data["gyr"][0:calib_s],
                                    mag_calib_data=imu_data["mag"][0:calib_s])

    # perform filter fusion on trial data to obtain segment orientations
    for idx, (acc, gyr, mag) in enumerate(zip(imu_data["acc"][calib_s:],
                                              imu_data["gyr"][calib_s:],
                                              imu_data["mag"][calib_s:])):
        pred_ori = ffilts.update(acc=acc, gyr=gyr, mag=mag)
        assert pred_ori.shape == (17, 4)
