
import numpy as np

from .base_skeleton import Skeleton, r, b, g, gr
from hipose.rotations import quat_mult, convert_euler_to_quat


xsens_segment_names = (
        "pelvis", "l5", "l3", "t12", "t8",                                                  # 0-4
        "neck", "head", "right_shoulder", "right_upper_arm", "right_forearm",               # 5-9
        "right_hand", "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",       # 10-14
        "right_upper_leg", "right_lower_leg", "right_foot", "right_toe", "left_upper_leg",  # 15-19
        "left_lower_leg", "left_foot", "left_toe"                                           # 20-22
)


class SkeletonXsens(Skeleton):
    """
    Define a Skeleton structure like the one used by Xsens, containing:
    (28 keypoints), (23 joints) and (23 segments).
    """
    def __init__(self, segment_lengths=None, ref_angles="tpose"):
        joint_names = [
                "pelvis", "l5", "l3", "t12", "t8",                                                  # [0, 1, 2, 3, 4]
                "neck", "head", "right_shoulder", "right_upper_arm", "right_forearm",               # [5, 6, 7, 8, 9]
                "right_hand", "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",       # [10,11,12,13,14]
                "right_upper_leg", "right_lower_leg", "right_foot", "right_toe", "left_upper_leg",  # [15,16,17,18,19]
                "left_lower_leg", "left_foot", "left_toe",                                          # [20,21,22]
        ]

        end_points = ["(top_head)",                                                                  # [23]
                      "(right_hand_finger_tips)", "(left_hand_finger_tips)",                         # [24,25]
                      "(right_foot_finger_tips)", "(left_foot_finger_tips)"                          # [26,27]
        ]

        # points of interest
        keypoints = [*joint_names, *end_points]

        root_joint = 0
        keypoint_colors = None

        #              |   pelvis->head(7)     |   right arm(5)   |    left arm(5)      |    right leg(5)    |   left leg(5)
        seg_start_pts = [0, 1, 2, 3, 4, 5, 6,    5,  7, 8, 9,  10,   5,  11, 12, 13, 14,   0,  15, 16, 17, 18,   0,  19, 20, 21, 22]
        seg_end_pts   = [1, 2, 3, 4, 5, 6, 23,   7,  8, 9, 10, 24,   11, 12, 13, 14, 25,   15, 16, 17, 18, 26,   19, 20, 21, 22, 27]
        seg_colors    = [g, g, g, g, g, g, g,    gr, r, r, r,  r,    gr, b,  b,  b,  b,    gr, r,  r,  r,  r,    gr, b,  b,  b,  b]
        seg_names     = ["pelvis", "l5", "l3", "t12", "t8", "neck", "head",
                         "(right_scapular)", "right_shoulder",  "right_upper_arm", "right_forearm", "right_hand",
                         "(left_scapular)",  "left_shoulder",   "left_upper_arm",  "left_forearm",  "left_hand",
                         "(right_pelvis)",   "right_upper_leg", "right_lower_leg", "right_foot",    "right_toe",
                         "(left_pelvis)",    "left_upper_leg",  "left_lower_leg",  "left_foot",     "left_toe",
                         ]

        # default segment lengths
        seg_lengths = (np.array([0.10, 0.10, 0.10, 0.15, 0.10, 0.05, 0.05,
                                 0.05, 0.10, 0.35, 0.25, 0.10,
                                 0.05, 0.10, 0.35, 0.25, 0.10,
                                 0.10, 0.45, 0.43, 0.10, 0.06,
                                 0.10, 0.45, 0.43, 0.10, 0.06])
                       if segment_lengths is None else segment_lengths)

        if ref_angles == "tpose":
            # default segment angles
            default_seg_angles = convert_euler_to_quat(
                    [[0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],

                     # [0, -np.pi / 2, -np.pi / 6],
                     [0, 0, -np.pi / 2],
                     [0, 0, -np.pi / 2],
                     [0, 0, -np.pi / 2],
                     [0, 0, -np.pi / 2],

                     # [0, -np.pi / 2, np.pi / 6],
                     [0, 0, np.pi / 2],
                     [0, 0, np.pi / 2],
                     [0, 0, np.pi / 2],
                     [0, 0, np.pi / 2],

                     # [0, 0, -np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0],
                     [0, 0, 0],

                     # [0, 0, np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    seq="XYZ")

        elif ref_angles == "npose":
            default_seg_angles = convert_euler_to_quat(
                    [[0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],
                     [0, -np.pi / 2, 0],

                     # [0, -np.pi / 2, -np.pi / 6],
                     [0, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],

                     # [0, -np.pi / 2, np.pi / 6],
                     [0, 0, np.pi / 2],
                     [-np.pi / 2, 0, np.pi / 2],
                     [-np.pi / 2, 0, np.pi / 2],
                     [-np.pi / 2, 0, np.pi / 2],

                     # [0, 0, -np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0],
                     [0, 0, 0],

                     # [0, 0, np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    seq="XYZ"
            )

        elif ref_angles == "glob_seg":
            # already in visualization mode (x-axis pointing to next segment)
            default_seg_angles = np.zeros((23, 4)) + [1, 0, 0, 0]
        else:
            raise NotImplementedError(f"Invalid referential specified ({ref_angles}), "
                                      f"choose one of ['glob_seg', 'tpose', 'npose']")

        #################################################################

        super(SkeletonXsens, self).__init__(root_joint=root_joint,
                                            joint_names=keypoints,
                                            joint_colors=keypoint_colors,
                                            segment_starts=seg_start_pts,
                                            segment_ends=seg_end_pts,
                                            segment_colors=seg_colors,
                                            segment_names=seg_names,
                                            segment_lengths=seg_lengths,
                                            segment_angles=default_seg_angles)

    def add_dummy_segments(self, segments):
        orient_r_scapula = quat_mult(
                segments[..., [4], :],
                convert_euler_to_quat([0, -np.pi / 2, (-np.pi / 1.8)], seq="XYZ"))
        orient_l_scapula = quat_mult(
                segments[..., [4], :],
                convert_euler_to_quat([0, -np.pi / 2, (np.pi / 1.8)], seq="XYZ"))
        orient_r_hip = quat_mult(
                segments[..., [0], :],
                convert_euler_to_quat([0, 0, -np.pi / 2], seq="XYZ"))
        orient_l_hip = quat_mult(
                segments[..., [0], :],
                convert_euler_to_quat([0, 0, np.pi / 2], seq="XYZ"))

        orientations_with_dummy = np.concatenate(
                [segments[..., 0:7, :],
                 orient_r_scapula, segments[..., 7:11, :],
                 orient_l_scapula, segments[..., 11:15, :],
                 orient_r_hip, segments[..., 15:19, :],
                 orient_l_hip, segments[..., 19:23, :],
                 ],
                axis=-2)
        return orientations_with_dummy

    def remove_dummy_segments(self, segments):
        nodummy_idx = [0, 1, 2, 3, 4, 5, 6,
                       8, 9, 10, 11,            # 7
                       13, 14, 15, 16,          # 12
                       18, 19, 20, 21,          # 17
                       23, 24, 25, 26]          # 22
        if len(np.shape(segments)) == 1:
            return segments[nodummy_idx]
        else:
            return segments[..., nodummy_idx, :]


class SkeletonXsensUpper(Skeleton):
    # Define a Skeleton structure like the one used by Xsens without lower body
    # (17 keypoints), (15 joints) and (15 segments).
    def __init__(self, segment_lengths=None, ref_angles="tpose"):
        joint_names = [
            "pelvis", "l5", "l3", "t12", "t8",                                   # [0, 1, 2, 3, 4]
            "neck", "head",                                                      # [5, 6]
            "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",  # [7, 8, 9, 10]
            "left_shoulder",  "left_upper_arm",  "left_forearm",  "left_hand",   # [11, 12, 13, 14]
        ]

        end_points = [
            "(top_head)",                                                         # [15]
            "(right_hand_finger_tips)", "(left_hand_finger_tips)",                # [16, 17]
        ]

        # points of interest
        keypoints = [*joint_names, *end_points]

        root_joint = 0
        keypoint_colors = None

        #              | pelvis->head |  right arm  |  left arm
        seg_start_pts = [0, 1, 2, 3, 4, 5, 6,    5,  7, 8, 9,  10,    5,  11, 12, 13, 14]
        seg_end_pts   = [1, 2, 3, 4, 5, 6, 15,   7,  8, 9, 10, 16,    11, 12, 13, 14, 17]
        seg_colors    = [g, g, g, g, g, g, g,    gr, r, r, r,  r,     gr,  b,  b,  b, b]
        seg_names     = ["pelvis", "l5", "l3", "t12", "t8", "neck", "head",
                         "(right_scapular)", "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",
                         "(left_scapular)",  "left_shoulder",  "left_upper_arm",  "left_forearm",  "left_hand"]

        # default segment lengths
        seg_lengths = (np.array([0.10, 0.10, 0.10, 0.15, 0.10, 0.05, 0.05,
                                 0.05, 0.10, 0.35, 0.25, 0.10,
                                 0.05, 0.10, 0.35, 0.25, 0.10])
                       if segment_lengths is None else segment_lengths)

        if ref_angles == "tpose":
            default_seg_angles = convert_euler_to_quat(
                    [[-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],

                     [0, 0, -np.pi/2],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],

                     [0, 0, np.pi/2],
                     [np.pi / 2, 0, np.pi / 2],
                     [np.pi / 2, 0, np.pi / 2],
                     [np.pi / 2, 0, np.pi / 2]],
                    seq="YXZ")

        elif ref_angles == "npose":
            default_seg_angles = convert_euler_to_quat(
                    [[-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],

                     [0, 0, -np.pi/2],
                     [np.pi / 2, np.pi / 2, 0],
                     [np.pi / 2, np.pi / 2, 0],
                     [np.pi / 2, np.pi / 2, 0],

                     [0, 0, np.pi / 2],
                     [np.pi / 2, -np.pi / 2, 0],
                     [np.pi / 2, -np.pi / 2, 0],
                     [np.pi / 2, -np.pi / 2, 0]],
                    seq="YXZ")

        elif ref_angles == "glob_seg":
            # already in visualization model (x-axis pointing to next segment)
            default_seg_angles = np.zeros((15, 4)) + [1, 0, 0, 0]
        else:
            raise NotImplementedError(f"Invalid referential specified ({ref_angles}), "
                                      f"choose one of ['glob_seg', 'tpose', 'npose']")

        ################################################################

        super(SkeletonXsensUpper, self).__init__(root_joint=root_joint,
                                                 joint_names=keypoints,
                                                 joint_colors=keypoint_colors,
                                                 segment_starts=seg_start_pts,
                                                 segment_ends=seg_end_pts,
                                                 segment_colors=seg_colors,
                                                 segment_names=seg_names,
                                                 segment_lengths=seg_lengths,
                                                 segment_angles=default_seg_angles)

    def add_dummy_segments(self, segments):
        orient_r_scapula = quat_mult(
                segments[..., [4], :],
                convert_euler_to_quat([0, 0, (-np.pi / 1.8)], seq="YXZ"))
        orient_l_scapula = quat_mult(
                segments[..., [4], :],
                convert_euler_to_quat([0, 0, (np.pi / 1.8)], seq="YXZ"))

        orientations_with_dummy = np.concatenate(
                [segments[..., 0:7, :],
                 orient_r_scapula, segments[..., 7:11, :],
                 orient_l_scapula, segments[..., 11:15, :],
                 ], axis=-2)
        return orientations_with_dummy

    def remove_dummy_segments(self, segments):
        nodummy_idx = [0, 1, 2, 3, 4, 5, 6,
                       8, 9, 10, 11,            # 7
                       13, 14, 15, 16]          # 12

        if len(np.shape(segments)) == 1:
            return segments[nodummy_idx]
        else:
            return segments[..., nodummy_idx, :]
