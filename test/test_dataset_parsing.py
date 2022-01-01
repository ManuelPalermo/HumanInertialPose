
import pytest
import numpy as np
import warnings

from hipose.utils import find_resource_path

warnings.filterwarnings("ignore")


def test_XsensAnalyse_parsing():
    from hipose.data.trial_parsing import extract_xsens_analyse_data

    trial_path = find_resource_path("./resources/sample_trials/ergowear/example_calibration_xsens*")
    data = extract_xsens_analyse_data(trial_path, resample_freq=30.,
                                      reset_position=True, reset_orientation=True)

    assert data["num_samples"] > 500
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            assert len(v) == data["num_samples"]


def test_XsensMtManager_parsing():
    from hipose.data.trial_parsing import extract_mtmanager_data

    trial_path = find_resource_path("./resources/sample_trials/mtwawinda/example_calibration_xsens*")
    data = extract_mtmanager_data(trial_path, ignore_mag=True,
                                  resample_freq=30.,
                                  orient_algorithm="madgwick",
                                  remove_g_vec=True)

    assert data["num_samples"] > 100
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            assert len(v) == data["num_samples"]


def test_XsensMTwAwinda_parsing():
    from hipose.data.trial_parsing import extract_mtwawinda_data

    sensorID_to_segmentName = {
            "00B45C23": "right_hand",
            "00B45C24": "right_foot",
            "00B45C25": "left_shoulder",
            "00B45C26": "left_upper_arm",
            "00B45C27": "left_forearm",
            "00B45C28": "left_hand",
            "00B45C29": "head",
            "00B45C2A": "stern",
            "00B45C2B": "right_upper_leg",
            "00B45C2C": "left_upper_leg",
            "00B45C2D": "pelvis",
            "00B45C2E": "right_lower_leg",
            "00B45C2F": "right_shoulder",
            "00B45C30": "right_forearm",
            "00B45C31": "left_foot",
            "00B45C33": "left_lower_leg",
            "00B45C34": "right_upper_arm",
    }

    trial_path = find_resource_path("./resources/sample_trials/mtwawinda/example_calibration_xsens*")
    data = extract_mtwawinda_data(trial_path,
                                  mapping_sensorid_to_segname=sensorID_to_segmentName,
                                  ignore_mag=True,
                                  resample_freq=30.,
                                  orient_algorithm="madgwick",
                                  remove_g_vec=True)

    assert data["num_samples"] > 100
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            assert len(v) == data["num_samples"]


@pytest.mark.filterwarnings("ignore")
def test_Ergowear_parsing():
    from hipose.data.trial_parsing import extract_ergowear_data

    trial_path = find_resource_path("./resources/sample_trials/ergowear/example_calibration_xsens*")
    data = extract_ergowear_data(trial_path, ignore_mag=True,
                                 resample_freq=30.,
                                 orient_algorithm="madgwick",
                                 remove_g_vec=True,
                                 ref_angles="tpose")

    assert data["num_samples"] > 100
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            assert len(v) == data["num_samples"]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("freq", [15., 60.])
@pytest.mark.parametrize("ref_angles", ["npose", "tpose"])
@pytest.mark.parametrize("s2s_calib_method", ["static", "static_mag"])
def test_Ergowear_error(freq, ref_angles, s2s_calib_method):
    from hipose.data.dataset_parsing.parse_cip_ergowear_dataset import map_segs_xsens2ergowear
    from hipose.data.trial_parsing import extract_ergowear_data, extract_xsens_analyse_data
    from hipose.utils import truncate_dict_of_arrays
    from hipose.metrics import qad

    trial_path = find_resource_path("./resources/sample_trials/ergowear/example_calibration_xsens*")
    ergowear_data = extract_ergowear_data(trial_path, ignore_mag=True,
                                          resample_freq=freq,
                                          orient_algorithm="madgwick",
                                          remove_g_vec=False,
                                          ref_angles=ref_angles,
                                          s2s_calib_method=s2s_calib_method,
                                          plot_data=False)

    xsens_data = extract_xsens_analyse_data(trial_path, ref_angles=ref_angles,
                                            resample_freq=freq,
                                            reset_position=True,
                                            reset_orientation=True,
                                            plot_data=False)

    # discard the initial calibration samples (based on ergowear segmentation algorithm)
    ergowear_data = truncate_dict_of_arrays(ergowear_data,
                                            s_idx=ergowear_data["trial_start_idx"])
    xsens_data = truncate_dict_of_arrays(xsens_data,
                                         s_idx=ergowear_data["trial_start_idx"])
    ergowear_data["num_samples"] -= ergowear_data["trial_start_idx"]
    xsens_data["num_samples"] -= ergowear_data["trial_start_idx"]

    qad_counter = []
    for i, (ori_xsens, ori_ergowear) in enumerate(zip(xsens_data['segments_quat'],
                                                       ergowear_data['segments_quat'])):
        # map xsens data to ergowear
        ori_xlikeergo = map_segs_xsens2ergowear(ori_xsens)
        qad_counter.append(qad(ori_ergowear, ori_xlikeergo))

    qad_metric = np.rad2deg(np.mean(qad_counter))
    assert qad_metric < 30         # less than 30º


@pytest.mark.filterwarnings("ignore")
def test_ErgowearDatabase_parsing():
    from hipose.data.dataset_parsing import parse_cip_ergowear_dataset
    from hipose.data.dataset_parsing import report_cip_ergowear_database_status

    dataset_path = find_resource_path("./resources/sample_trials/ergowear/")
    subj_ids = ["example"]
    seq_ids = ["task", "circuit", "sequence", "random", "validation", "calibration"]
    rep_ids = [f"rep{i}" for i in range(1, 10)]
    rep_ids_val = [f"t{i:02d}" for i in range(1, 45)]
    rep_ids_calib = ["xsens"]

    report_cip_ergowear_database_status(dataset_path, verbose=False, save_path=None)
    parse_cip_ergowear_dataset(dataset_path,
                           subject_ids=subj_ids,
                           sequence_ids=seq_ids,
                           repetition_ids=rep_ids + rep_ids_val + rep_ids_calib,
                           resample_freq=30.,
                           save_path=None,
                           ref_angles="tpose",
                           s2s_calib_method="optim",
                           verbose=False,
                           ignore_errors=False,
                           xsens_kwargs=dict(add_g_vec=True),
                           ergowear_kwargs=dict(orient_algorithm="madgwick",
                                                ignore_mag=True,
                                                remove_g_vec=False)
                           )


def test_MTwAwindaDatabase_parsing():
    from hipose.data.dataset_parsing import parse_cip_mtwawinda_dataset
    from hipose.data.dataset_parsing import report_cip_mtwawinda_database_status

    dataset_path = find_resource_path("./resources/sample_trials/mtwawinda/")

    subj_ids = ["example"]
    seq_ids = ["task", "circuit", "sequence", "random", "validation", "calibration"]
    rep_ids = [f"rep{i}" for i in range(1, 10)]
    rep_ids_calib = ["npose", "tpose", "frontpose", "sideways", "xsens"]

    report_cip_mtwawinda_database_status(dataset_path, verbose=False, save_path=None)
    parse_cip_mtwawinda_dataset(dataset_path,
                                subject_ids=subj_ids,
                                sequence_ids=seq_ids,
                                repetition_ids=rep_ids + rep_ids_calib,
                                resample_freq=30.,
                                save_path=None,
                                ref_angles="tpose",
                                s2s_calib_method="optim",
                                verbose=False,
                                ignore_errors=False,
                                xsens_kwargs=dict(add_g_vec=True),
                                mtwawinda_kwargs=dict(orient_algorithm="ekf",
                                                      ignore_mag=False,
                                                      remove_g_vec=False)
                                )


def test_TotalCapture_parsing():
    pass


def test_DIPIMU_parsing():
    pass
