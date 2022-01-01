import os
import re
import time
import numpy as np

from scipy.constants import g

from hipose.rotations import convert_quat_to_euler, qad
from hipose.data.trial_parsing import extract_mtwawinda_data, extract_xsens_analyse_data

from hipose.utils import truncate_dict_of_arrays, nested_dict, \
    convert_nestedddict_to_regular

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


def map_segs_xsens2mtwawinda(xsens_segs):
    """Map segments from xsens (23) to mtwawinda (17)."""
    return xsens_segs[..., [0, 4, 6,
                            7, 8, 9, 10,
                            11, 12, 13, 14,
                            15, 16, 17,
                            19, 20, 21], :]


def cip_mtwawinda_database_iterator(dataset_root_path, subject_ids=None,
                                    sequence_ids=None, repetition_ids=None):
    """
    Iterates over all mtwawinda trials inside a directory, yielding the
    corresponding path and some additional information.

    Args:
        dataset_root_path(str): directory to search and extract
            mtwawinda trials data.
        subject_ids(list(str), None): ids of the subjects to extract.
            Or None to yield all.
        sequence_ids(list(str), None): ids of the sequences to extract.
            Or None to yield all.
        repetition_ids(list(str), None): ids of the repetitions to
            yield. Or None to yield all.

    """
    # pattern to match trial folder names
    pat_subj = "subject[0-9]{2}"
    pat_seq = "(task|circuit|sequence|random|calibration)"
    pat_rep = "(rep[0-9]|npose|tpose|frontpose|sideways|xsens)"
    trial_match_pattern = re.compile(f"{pat_subj}_{pat_seq}_{pat_rep}.*")

    for subdir, dirs, files in sorted(os.walk(dataset_root_path)):
        # trial folder path
        trial_id = os.path.basename(subdir)

        # extract trial information from file names
        trial_match = re.match(trial_match_pattern, trial_id)
        if trial_match is not None:
            subj_id, seq_id, rep_id, *extra_info = trial_match.group(0).split("_")
            extra_info = "_".join(extra_info)
        else:
            subj_id = seq_id = rep_id = extra_info = None

        if (((subject_ids is None) or subj_id in subject_ids)
                and (sequence_ids is None or seq_id in sequence_ids)
                and (repetition_ids is None or rep_id in repetition_ids)
                and (subj_id is not None) and (seq_id is not None)
                and (rep_id is not None)):
            yield subdir, trial_id, subj_id, seq_id, rep_id, extra_info


def parse_cip_mtwawinda_dataset(dataset_root_path, subject_ids, sequence_ids, repetition_ids,
                                resample_freq=60., save_path="./CIP/MTwAwinda_processed/",
                                ref_angles="npose", s2s_calib_method="optim",
                                verbose=True, ignore_errors=True,
                                mtwawinda_kwargs=None, xsens_kwargs=None):
    """
    Parse all mtwawinda trials inside "dataset_root_path".

    Args:
        dataset_root_path(str): directory to search and extract
            trials data.
        subject_ids(list(str)): ids of the subjects to extract.
        sequence_ids(list(str)): ids of the sequences to extract.
        repetition_ids(list(str)): ids of the repetitions to extract.
        resample_freq(float): desired output signal frequency.
        save_path(save_path|str): path to save the extracted data, or
            None if data should not be saved (e.g. when testing).
        ref_angles(str): referential to return segment angles. Can be
            one of "npose" or "tpose". Xsens uses "tpose" by
            default.
        s2s_calib_method(None|str): sensor to segment
            calibration method to be used on mtwawinda data. Can
            be [None, "manual", "static", "static_mag"].
        verbose(bool): if info should be printed to terminal.
        ignore_errors(bool): if should ignore errors when parsing
            dataset. These might occur from file problems in a
            single trial.
        mtwawinda_kwargs(dict[str, any]): dictionary of parameters to
            pass to the function which extract mtwawinda trial data.
        xsens_kwargs(dict[str, any]): dictionary of parameters to pass
            to the function which extract xsens trial data:
            ("data_utils.dataset_parsing.extract_xsens_data").

    """
    if mtwawinda_kwargs is None: mtwawinda_kwargs = dict()
    if xsens_kwargs is None: xsens_kwargs = dict()

    # create nested dicts to hold data
    imus_acc_dict = nested_dict()
    imus_gyr_dict = nested_dict()
    imus_mag_dict = nested_dict()
    imus_ori_dict = nested_dict()
    segs_ori_dict = nested_dict()
    segs_pos_dict = nested_dict()
    jnts_ang_dict = nested_dict()
    root_pos_dict = nested_dict()
    root_yaw_dict = nested_dict()
    com_pos_dict  = nested_dict()
    metadata_dict = dict(dataset="MTwAwinda", subjects=set(), sequences=set(), repetitions=set(),
                         modalities=["imus_acc", "imus_gyr", "imus_mag", "imus_ori", "segs_ori",
                                     "segs_pos", "jnts_ang", "root_pos", "root_yaw", "com_pos"],
                         num_samples=0, n_segments=17, freq=resample_freq, ref_angles=ref_angles,
                         mtwawinda_parsing_kwargs={**mtwawinda_kwargs,
                                                   "s2s_calib_method": s2s_calib_method},
                         xsens_parsing_kwargs=xsens_kwargs)

    n_extracted_samples = 0
    if verbose: print("Extracting data from directory, this will take a while!")
    for subdir, trial_id, subj_id, seq_id, rep_id, extra_info \
            in cip_mtwawinda_database_iterator(dataset_root_path, subject_ids,
                                               sequence_ids, repetition_ids):

        try:
            t_start = time.perf_counter()

            if s2s_calib_method == "optim":
                import json
                with open(dataset_root_path + "/s2s_calibration_optim.json") as jcalib:
                    s2s_calib = json.load(jcalib)
                subj_s2s_calib = np.array(s2s_calib[subj_id])
                mt_s2s_calib_method = "manual"
            else:
                subj_s2s_calib = "default"
                mt_s2s_calib_method = s2s_calib_method

            # extract data from mtwawinda files
            mtwawinda_data = extract_mtwawinda_data(
                    subdir, mapping_sensorid_to_segname=sensorID_to_segmentName,
                    resample_freq=resample_freq,
                    ref_angles=ref_angles,
                    s2s_calib_method=mt_s2s_calib_method,
                    imus_manual_alignment=subj_s2s_calib,
                    **mtwawinda_kwargs)

            # extract data from xsens files
            xsens_data = extract_xsens_analyse_data(
                    subdir, resample_freq=resample_freq,
                    ref_angles=ref_angles,
                    reset_position=True, reset_orientation=True,
                    initial_heading=mtwawinda_data["initial_heading"],
                    **xsens_kwargs)

            # if only less than 5s of data exist, then discard trial
            # (rest has probably been corrupted)
            if ((mtwawinda_data["num_samples"] < 5)
                    or (mtwawinda_data["timestamps"][-1] < 5.0)
                    or (xsens_data["timestamps"][-1] < 5.0)):
                if verbose:
                    print(f"Trial has been discarded as it only contains "
                          f"{mtwawinda_data['num_samples']} samples.")
                continue

            # remove calibration samples + truncate xsens data to num mtwawinda samples
            n_samples = min(mtwawinda_data["num_samples"], xsens_data["num_samples"])
            xsens_data = truncate_dict_of_arrays(xsens_data,
                                                 s_idx=mtwawinda_data["trial_start_idx"],
                                                 e_idx=n_samples)
            mtwawinda_data = truncate_dict_of_arrays(mtwawinda_data,
                                                     s_idx=mtwawinda_data["trial_start_idx"],
                                                     e_idx=n_samples)
            mtwawinda_data["num_samples"] -= mtwawinda_data["trial_start_idx"]
            xsens_data["num_samples"] = mtwawinda_data["num_samples"]

            # process segment angles
            xlikemtwawinda_seg_quats     = map_segs_xsens2mtwawinda(xsens_data["segments_quat"])
            xlikemtwawinda_seg_pos       = map_segs_xsens2mtwawinda(xsens_data["segments_pos"])
            xlikemtwawinda_joints_angles = map_segs_xsens2mtwawinda(xsens_data["joints_angle"])

            # extract root heading (where the skeleton is facing in the world)
            xlikemtwawinda_root_yaw = convert_quat_to_euler(
                    xlikemtwawinda_seg_quats[:, 0], seq="xyz")[:, None, [2]]

            # transform root data to delta(Δ) (displacement at every timestep - odometry)
            root_pos_delta = np.concatenate([np.zeros((1, 1, 3)),
                                             np.diff(xsens_data["segments_pos"][:, [0], :], axis=0)
                                             ], axis=0)
            root_yaw_delta = np.concatenate([np.zeros((1, 1, 1)),
                                             np.diff(xlikemtwawinda_root_yaw,  axis=0)
                                             ], axis=0)

            if save_path is not None:
                # add trial data to nested dict
                imus_acc_dict[subj_id][seq_id][rep_id] = (mtwawinda_data["acc"] / g).astype(np.float32)
                imus_gyr_dict[subj_id][seq_id][rep_id] = mtwawinda_data["gyr"].astype(np.float32)
                imus_mag_dict[subj_id][seq_id][rep_id] = mtwawinda_data["mag"].astype(np.float32)
                imus_ori_dict[subj_id][seq_id][rep_id] = mtwawinda_data["segments_quat"].astype(np.float32)
                segs_ori_dict[subj_id][seq_id][rep_id] = xlikemtwawinda_seg_quats.astype(np.float32)
                segs_pos_dict[subj_id][seq_id][rep_id] = xlikemtwawinda_seg_pos.astype(np.float32)
                jnts_ang_dict[subj_id][seq_id][rep_id] = xlikemtwawinda_joints_angles.astype(np.float32)
                root_pos_dict[subj_id][seq_id][rep_id] = root_pos_delta.astype(np.float32)
                root_yaw_dict[subj_id][seq_id][rep_id] = root_yaw_delta.astype(np.float32)
                com_pos_dict[subj_id][seq_id][rep_id]  = xsens_data["center_of_mass"].astype(np.float32)
                # add general metadata (could be computed only once - just ignoring repeating)
                metadata_dict["subjects"].add(subj_id)
                metadata_dict["sequences"].add(seq_id)
                metadata_dict["repetitions"].add(rep_id)

            # calculate trial parsing error
            trial_avg_error = np.mean(
                    qad(mtwawinda_data["segments_quat"], xlikemtwawinda_seg_quats)
            )

            t_end = round(time.perf_counter() - t_start, 1)
            if verbose: print(f"[{t_end}s]   "
                              f"Extracted ({mtwawinda_data['num_samples']}) samples from trial:   {trial_id}  "
                              f"    |    AvgError: {np.round(np.rad2deg(trial_avg_error), 3)}º")
            n_extracted_samples += mtwawinda_data["num_samples"]

        except Exception as err:
            if not ignore_errors: raise err
            if verbose: print(f"Could not extract data from: {trial_id}    |    {err}")

    if save_path is not None:
        # create dir to save data if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # prepare metadata for save
        metadata_dict["num_samples"] = n_extracted_samples
        for k, v in metadata_dict.items():
            if isinstance(v, set):
                metadata_dict[k] = list(v)

        # sve extracted data as compressed .npy array
        save_file = save_path + "/MTwAwindaDataset" + ".npz"
        np.savez(save_file,
                 imus_acc=convert_nestedddict_to_regular(imus_acc_dict),
                 imus_gyr=convert_nestedddict_to_regular(imus_gyr_dict),
                 imus_mag=convert_nestedddict_to_regular(imus_mag_dict),
                 imus_ori=convert_nestedddict_to_regular(imus_ori_dict),
                 segs_ori=convert_nestedddict_to_regular(segs_ori_dict),
                 segs_pos=convert_nestedddict_to_regular(segs_pos_dict),
                 jnts_ang=convert_nestedddict_to_regular(jnts_ang_dict),
                 root_pos=convert_nestedddict_to_regular(root_pos_dict),
                 root_yaw=convert_nestedddict_to_regular(root_yaw_dict),
                 com_pos=convert_nestedddict_to_regular(com_pos_dict),
                 metadata=metadata_dict,
                 )
        if verbose: print("\nSaved extracted data to: ", save_file)
    if verbose: print(f"Extracted a total of {n_extracted_samples} samples from Ergowear dataset!")


def report_cip_mtwawinda_database_status(dataset_root_path, subject_ids=None,
                                         sequence_ids=None, repetition_ids=None,
                                         save_path="./",
                                         verbose=False):
    """
    Inspect all MTwAwinda trials inside directory.

    Args:
        dataset_root_path(str): directory to search and extract
            trials data.
        subject_ids(list(str), None): ids of the subjects to extract.
            Or None to yield all.
        sequence_ids(list(str), None): ids of the sequences to extract.
            Or None to yield all.
        repetition_ids(list(str), None): ids of the repetitions to
            yield. Or None to yield all.
        save_path(str, None): path to save report.
        verbose(bool): if problems should be printed to terminal.

    """
    import pandas as pd
    from hipose.data.trial_parsing import report_mtmanager_trial_status

    dataset_report = pd.DataFrame(columns=["trial_path", "num_samples", "path_name",
                                           "lost_samples", "sampl_freq",
                                           "match_xsens_files", "data_modalities",
                                           "sensor_ids"])

    if verbose:
        print("Analysing MTwAwinda database for trial problems, this will "
              "take a while and a report will be generated at the end!")

    for subdir, trial_id, *_ in cip_mtwawinda_database_iterator(dataset_root_path, subject_ids,
                                                                sequence_ids, repetition_ids):

        try:
            report = report_mtmanager_trial_status(subdir, verbose=verbose)
        except Exception as err:
            report = dict(trial_path=trial_id,
                          num_samples=np.nan,
                          path_name=np.nan,
                          lost_samples=np.nan,
                          sampl_freq=np.nan,
                          match_xsens_files=np.nan,
                          data_modalities=np.nan,
                          sensor_ids=np.nan)
            if verbose:
                print("\nCould not load data from:  ", trial_id, "   |   ", err)
        dataset_report = dataset_report.append(report, ignore_index=True)

    if verbose:
        print("\n", dataset_report)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        dataset_report.to_csv(save_path + "/MTwAwindaDatabaseStatusReport.csv")
