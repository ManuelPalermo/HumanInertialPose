
import time
import numpy as np

from hipose.data.trial_parsing.extract_ergowear import extract_ergowear_raw_data, default_ergowear_imus_manual_alignment
from hipose.data.dataset_parsing.parse_cip_ergowear_dataset import map_segs_xsens2ergowear
from hipose.data.trial_parsing.extract_xsens_analyse import extract_xsens_analyse_data
from hipose.skeleton import SkeletonXsens, SkeletonErgowear, SkeletonVisualizer


def compute_and_evaluate_inertial_pose(example_data_path):
    # initialize raw inertial data (ergowear dataset example sample)
    imu_data = extract_ergowear_raw_data(
            example_data_path,
            interpolate_missing=True     # interpolate missing samples to synchronize with GT
    )

    # initialize gt skeleton data (from synchronized Xsens)
    gt_data = extract_xsens_analyse_data(
            example_data_path, ref_angles="npose",          # reference s2s calibration will be in NPose
            resample_freq=imu_data["freq"],                 # resample data to match desired freq
            reset_position=True, reset_orientation=True,    # center skeleton root (XY) on the world facing forward
    )

    # create skeletons for 3D visualization
    skel_pred = SkeletonErgowear(ref_angles="npose", segment_lengths=None)
    skel_gt = SkeletonXsens(ref_angles="npose", segment_lengths=None)
    vis = SkeletonVisualizer(dict(skel_gt=skel_gt,
                                  skel_pred=skel_pred),
                             display_segment_axis=True,         # turn off for faster rendering
                             animation_fps=imu_data["freq"])

    # define metrics to evaluate (between ergowear and xsens skeletons)
    from hipose.metrics import MetricsAnalyser, QADistMetric, AUCMetric, TimeMetric
    metrics_log = MetricsAnalyser(
            exp_name="ExampleMetrics",
            metrics=dict(
                    QuatAngleDistance=QADistMetric("QuatAngleDistance",
                                                   description="segment_angles",
                                                   err_thresh=np.pi / 6,
                                                   show_degrees=True),
                    auc_qad=AUCMetric("auc_qad", description="segment_angles",
                                      dist="qad", units="%",
                                      pcp_thresh_range=(0, np.pi)),
                    processing_time=TimeMetric("processing_time", units="ms"),
            ))

    # initialize filter fusion (example trial has 9 IMUs)
    from hipose.api.fusion_filter import InertialPoseFusionFilter
    ffilts = InertialPoseFusionFilter(
            num_imus=9,
            ignore_mag=True,
            fusion_filter_alg="madgwick",
            s2s_calib_method="static",
            default_data_freq=imu_data["freq"]
    )

    # initialize calibration params from static NPose
    # (example trial has 5s of NPose at the start)
    calib_s = int(imu_data["freq"] * 5)
    ffilts.compute_imus_calibration(acc_calib_data=imu_data["acc"][0:calib_s],
                                    gyr_calib_data=imu_data["gyr"][0:calib_s],
                                    mag_calib_data=imu_data["mag"][0:calib_s],
                                    manual_s2s_alignment=default_ergowear_imus_manual_alignment)

    # perform filter fusion on trial data to obtain segment orientations
    for idx, (acc, gyr, mag, gt_ori, root_pos) in enumerate(
            zip(imu_data["acc"][calib_s:],
                imu_data["gyr"][calib_s:],
                imu_data["mag"][calib_s:],
            gt_data["segments_quat"][calib_s:],
            gt_data["center_of_mass"][calib_s:])):
        start_time = time.perf_counter()

        # compute segment orientations
        pred_ori = ffilts.update(acc=acc, gyr=gyr, mag=mag)

        # select matching segments from GT
        mapgt_ori = map_segs_xsens2ergowear(gt_ori)

        # compute metrics between computed segment orientations and GT
        end_time = time.perf_counter() - start_time
        metrics_log.update(
                dict(QuatAngleDistance=[pred_ori, mapgt_ori],
                     auc_qad=[pred_ori, mapgt_ori],
                     processing_time=[end_time * 1000.],
                     )
        )

        # visualize motion in 3D (pred vs GT)
        vis.show3d(
                skeletons_orient_dict=dict(
                        skel_gt=gt_ori,
                        skel_pred=pred_ori),
                skeletons_root_pos=dict(
                        skel_gt=root_pos[0],
                        skel_pred=root_pos[0] + [0, 1.25, 0]),
        )

    # show computed metrics
    metrics_log.log_all(save_path=None, show_plots=True, print_metrics=True)


if __name__ == "__main__":
    compute_and_evaluate_inertial_pose(
            example_data_path="../resources/sample_trials/ergowear/example_calibration_xsens/"
    )
