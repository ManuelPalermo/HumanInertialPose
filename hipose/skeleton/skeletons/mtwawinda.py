
import numpy as np

from .base_skeleton import Skeleton, r, b, g, gr
from hipose.rotations import quat_mult, convert_euler_to_quat


mtwawinda_segment_names = (
        "pelvis", "stern", "head",
        "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",
        "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",
        "right_upper_leg", "right_lower_leg", "right_foot",
        "left_upper_leg", "left_lower_leg", "left_foot",
)


class SkeletonMTwAwinda(Skeleton):
    """
    Define a Skeleton structure like the one used by MTwAwinda, containing:
    (22 keypoints), (17 joints) and (17 segments).
    """
    def __init__(self, segment_lengths=None, ref_angles="tpose"):
        joint_names = [
                "pelvis", "stern", "head",                                              # [0, 1, 2]
                "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",     # [3, 4, 5, 6]
                "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",         # [7, 8, 9, 10]
                "right_upper_leg", "right_lower_leg", "right_foot",                     # [11, 12, 13]
                "left_upper_leg", "left_lower_leg", "left_foot",                        # [14, 15, 16]
        ]

        end_points = ["(top_head)",                                                     # [17]
                      "(right_hand_finger_tips)", "(left_hand_finger_tips)",            # [18, 19]
                      "(right_toes)", "(left_toes)"                                     # [20, 21]
        ]

        # points of interest
        keypoints = [*joint_names, *end_points]

        root_joint = 0
        keypoint_colors = None

        #              | pelvis->head(3) |  right arm(5)    |   left arm(5)     |  right leg(4)   |   left leg(4)
        seg_start_pts = [0, 1, 2,          2,  3, 4, 5, 6,    2,  7, 8, 9,  10,   0,  11, 12, 13,   0,  14, 15, 16]
        seg_end_pts   = [1, 2, 17,         3,  4, 5, 6, 18,   7,  8, 9, 10, 19,   11, 12, 13, 20,   14, 15, 16, 21]
        seg_colors    = [g, g, g,          gr, r, r, r, r,    gr, b, b, b,  b,    gr,  r, r,  r,    gr, b,  b,  b]
        seg_names     = ["lower_back", "upper_back", "neck",
                         "(right_scapular)", "right_shoulder", "right_upper_arm", "right_forearm", "right_hand",
                         "(left_scapular)",  "left_shoulder",  "left_upper_arm",  "left_forearm", "left_hand",
                         "(right_pelvis)", "right_upper_leg", "right_lower_leg", "right_foot",
                         "(left_pelvis)", "left_upper_leg", "left_lower_leg", "left_foot"]

        # default segment lengths
        seg_lengths = (np.array([0.30, 0.25, 0.10,
                                 0.05, 0.10, 0.35, 0.25, 0.10,
                                 0.05, 0.10, 0.35, 0.25, 0.10,
                                 0.10, 0.45, 0.43, 0.10,
                                 0.10, 0.45, 0.43, 0.10])
                       if segment_lengths is None else segment_lengths)

        if ref_angles == "tpose":
            # default segment angles
            default_seg_angles = convert_euler_to_quat(
                    [[0, -np.pi / 2, 0],
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

                     # [0, 0, np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0]],
                    seq="XYZ")

        elif ref_angles == "npose":
            default_seg_angles = convert_euler_to_quat(
                    [[0, -np.pi / 2, 0],
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

                     # [0, 0, np.pi / 2],
                     [0, np.pi / 2, 0],
                     [0, np.pi / 2, 0],
                     [0, 0, 0]],
                    seq="XYZ"
            )

        elif ref_angles == "glob_seg":
            # already in visualization mode (x-axis pointing to next segment)
            default_seg_angles = np.zeros((17, 4)) + [1, 0, 0, 0]
        else:
            raise NotImplementedError(f"Invalid referential specified ({ref_angles}), "
                                      f"choose one of ['glob_seg', 'tpose', 'npose']")

        #################################################################

        super(SkeletonMTwAwinda, self).__init__(root_joint=root_joint,
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
                segments[..., [1], :],
                convert_euler_to_quat([0, -np.pi / 2, (-np.pi / 1.8)], seq="XYZ"))
        orient_l_scapula = quat_mult(
                segments[..., [1], :],
                convert_euler_to_quat([0, -np.pi / 2, (np.pi / 1.8)], seq="XYZ"))
        orient_r_hip = quat_mult(
                segments[..., [0], :],
                convert_euler_to_quat([0, 0, -np.pi / 2], seq="XYZ"))
        orient_l_hip = quat_mult(
                segments[..., [0], :],
                convert_euler_to_quat([0, 0, np.pi / 2], seq="XYZ"))

        orientations_with_dummy = np.concatenate(
                [segments[..., 0:3, :],
                 orient_r_scapula, segments[..., 3:7, :],
                 orient_l_scapula, segments[..., 7:11, :],
                 orient_r_hip, segments[..., 11:14, :],
                 orient_l_hip, segments[..., 14:17, :],
                 ],
                axis=-2)
        return orientations_with_dummy

    def remove_dummy_segments(self, segments):
        nodummy_idx = [0, 1, 2,
                       4, 5, 6, 7,              # 3
                       9, 10, 11, 12,           # 8
                       14, 15, 16,              # 13
                       18, 19, 20]              # 17
        if len(np.shape(segments)) == 1:
            return segments[nodummy_idx]
        else:
            return segments[..., nodummy_idx, :]


# TODO: implement xsens MTwAwinda upper body skeleton
