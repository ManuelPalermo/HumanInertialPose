
import numpy as np


def parse_ergowear(dataset_path, save_path, sampling_freq=60.,
                   report_status=False, ignore_errors=True):
    """Parse all ergowear trials in the database."""
    from hipose.data.dataset_parsing import parse_cip_ergowear_dataset

    subj_ids = ["subject{0:02d}".format(i) for i in range(0, 21)]
    seq_ids = ["task", "circuit", "sequence", "random", "validation", "calibration"]
    rep_ids = [f"rep{i}" for i in range(1, 10)]
    rep_ids_val = [f"t{i:02d}" for i in range(1, 45)]
    rep_ids_calib = ["xsens"]

    if report_status:
        from hipose.data.dataset_parsing import report_ergowear_database_status
        report_ergowear_database_status(dataset_path, save_path=save_path, verbose=True)

    parse_cip_ergowear_dataset(dataset_path,
                               subject_ids=subj_ids,
                               sequence_ids=seq_ids,
                               repetition_ids=rep_ids + rep_ids_val + rep_ids_calib,
                               resample_freq=sampling_freq,
                               ref_angles="tpose",
                               s2s_calib_method="static",
                               save_path=save_path,
                               ignore_errors=ignore_errors,
                               xsens_kwargs=dict(add_g_vec=True),
                               ergowear_kwargs=dict(orient_algorithm="madgwick",
                                                    ignore_mag=False,
                                                    remove_g_vec=False)
                               )
    return save_path + "/ErgowearDataset.npz"


def parse_mtwawinda(dataset_path, save_path, sampling_freq=60.,
                    trial_type="full", report_status=False,
                    ignore_errors=True):
    """Parse all mtwawinda trials in the database."""
    from hipose.data.dataset_parsing import parse_cip_mtwawinda_dataset

    if trial_type == "default":
        dataset_path = dataset_path + "/trials/"
        subj_ids = ["subject{0:02d}".format(i) for i in range(0, 10)]

    elif trial_type == "extra_indoors":
        dataset_path = dataset_path + "/trials_extra_indoors/"
        subj_ids = ["subject{0:02d}".format(i) for i in range(0, 5)]

    elif trial_type == "extra_outdoors":
        dataset_path = dataset_path + "/trials_extra_outdoors/"
        subj_ids = ["subject{0:02d}".format(i) for i in range(0, 5)]

    else:
        raise NotImplementedError

    seq_ids = ["task", "circuit", "sequence", "random", "validation", "calibration"]
    rep_ids = [f"rep{i}" for i in range(1, 10)]
    rep_ids_calib = ["npose", "tpose", "frontpose", "sideways", "xsens"]

    if report_status:
        from hipose.data.dataset_parsing import report_mtwawinda_database_status
        report_mtwawinda_database_status(dataset_path, save_path=save_path, verbose=True)

    parse_cip_mtwawinda_dataset(dataset_path,
                                subject_ids=subj_ids,
                                sequence_ids=seq_ids,
                                repetition_ids=rep_ids + rep_ids_calib,
                                resample_freq=sampling_freq,
                                ref_angles="tpose",
                                s2s_calib_method="optim",
                                save_path=save_path,
                                ignore_errors=ignore_errors,
                                xsens_kwargs=dict(add_g_vec=True),
                                mtwawinda_kwargs=dict(orient_algorithm="madgwick",
                                                      ignore_mag=False,
                                                      remove_g_vec=False)
                                )
    return save_path + "/MTwAwindaDataset.npz"


def check_parsed_dataset(dataset_path):
    """Check data inside a dataset parsed to common structure."""
    dataset = np.load(dataset_path, allow_pickle=True)
    metadata = dataset["metadata"].item()
    for k, v in metadata.items():
        print(k, ":", v)
    return dataset, metadata


if __name__ == "__main__":
    dataset2parse = ["Ergowear", "MTwAwinda"][1]
    dataset_path = f"../resources/datasets/CIP/{dataset2parse}/"
    save_path = "../resources/parsed_datasets/CIP/"

    # parse params
    sampling_freq = 60.
    ignore_errors = True
    report_status = False

    if dataset2parse == "Ergowear":
        out_path = parse_ergowear(dataset_path, save_path,
                                  sampling_freq=sampling_freq,
                                  ignore_errors=ignore_errors,
                                  report_status=report_status)

    elif dataset2parse == "MTwAwinda":
        mtwawinda_trial_type = "default"  # default | extra_indoors | extra_outdoors
        save_path = save_path + "/mtwawinda/" + mtwawinda_trial_type
        out_path = parse_mtwawinda(dataset_path, save_path,
                                   sampling_freq=sampling_freq,
                                   ignore_errors=ignore_errors,
                                   report_status=report_status,
                                   trial_type=mtwawinda_trial_type)

    else:
        raise NotImplementedError("Only ['Ergowear' and 'MTwAwinda] dataset parsing "
                                  "have been implemented so far!")

    # checks data inside parsed dataset
    check_parsed_dataset(out_path)
