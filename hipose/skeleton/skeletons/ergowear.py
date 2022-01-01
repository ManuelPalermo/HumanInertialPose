
import numpy as np

from .base_skeleton import Skeleton, r, b, g, gr
from hipose.rotations import quat_mult, convert_euler_to_quat


ergowear_segment_names = (
        "pelvis", "t12", "neck",                                                # [0,1,2]
        "right_upper_arm", "right_forearm", "right_hand",                       # [3,4,5]
        "left_upper_arm", "left_forearm", "left_hand",                          # [6,7,8]
)


class SkeletonErgowear(Skeleton):
    # Define a Skeleton structure like the one used by Ergowear, containing:
    # (12 keypoints), (9 joints) and (9 segments).
    def __init__(self, segment_lengths=None, ref_angles="npose"):
        joint_names = [
            "pelvis", "t12", "neck",                                            # [0,1,2]
            "r_shoulder", "r_elbow", "r_wrist",                                 # [3,4,5]
            "l_shoulder", "l_elbow", "l_wrist",                                 # [6,7,8]
        ]

        end_points = [
            "(top_head)",                                                       # [9]
            "(right_hand_finger_tips)", "(left_hand_finger_tips)"               # [10,11]
        ]

        keypoints = [*joint_names, *end_points]

        root_joint = 0
        keypoint_colors = None

        #              | pelvis->head |  right arm  |  left arm
        seg_start_pts = [0, 1, 2,        2,  3, 4, 5,      2,  6, 7, 8]
        seg_end_pts   = [1, 2, 9,        3,  4, 5, 10,     6,  7, 8, 11]
        seg_colors    = [g, g, g,        gr, r, r, r,      gr, b, b, b]
        seg_names     = ["lower_back", "upper_back", "neck",
                         "(r_scapular)", "r_upper_arm", "r_forearm", "r_hand",
                         "(l_scapular)", "l_upper_arm", "l_forearm", "l_hand"]

        # default segment lengths
        seg_lengths = (np.array([0.30, 0.25, 0.10,
                                 0.15, 0.35, 0.25, 0.1,
                                 0.15, 0.35, 0.25, 0.1])
                       if segment_lengths is None else segment_lengths)

        if ref_angles == "tpose":
            default_seg_angles = convert_euler_to_quat(
                    [[-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],

                     # [-np.pi / 2,  0,       (-np.pi / 1.8)],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],
                     [np.pi / 2, 0, -np.pi / 2],

                     # [-np.pi / 2,  0,       (np.pi / 1.8)],
                     [np.pi / 2, 0, np.pi / 2],
                     [np.pi / 2, 0, np.pi / 2],
                     [np.pi / 2, 0, np.pi / 2]],
                    seq="YXZ"
            )

        elif ref_angles == "npose":
            default_seg_angles = convert_euler_to_quat(
                    [[-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],
                     [-np.pi / 2, 0, 0],

                     # [-np.pi / 2,  0,       (-np.pi / 1.8)],
                     [np.pi / 2, np.pi / 2, 0],
                     [np.pi / 2, np.pi / 2, 0],
                     [np.pi / 2, np.pi / 2, 0],

                     # [-np.pi / 2,  0,       (np.pi / 1.8)],
                     [np.pi / 2, -np.pi / 2, 0],
                     [np.pi / 2, -np.pi / 2, 0],
                     [np.pi / 2, -np.pi / 2, 0]],
                    seq="YXZ"
            )

        elif ref_angles == "glob_seg":
            # already in visualization model (x-axis pointing to next segment)
            default_seg_angles = np.zeros((9, 4)) + [1, 0, 0, 0]
        else:
            raise NotImplementedError(f"Invalid referential specified ({ref_angles}), "
                                      f"choose one of ['glob_seg', 'tpose', 'npose']")

        ################################################################

        super(SkeletonErgowear, self).__init__(root_joint=root_joint,
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
                convert_euler_to_quat([0, 0, (-np.pi / 1.8)], seq="YXZ"))
        orient_l_scapula = quat_mult(
                segments[..., [1], :],
                convert_euler_to_quat([0, 0, (np.pi / 1.8)], seq="YXZ"))

        orientations_with_dummy = np.concatenate(
                [segments[..., [0, 1, 2], :],
                 orient_r_scapula, segments[..., [3, 4, 5], :],
                 orient_l_scapula, segments[..., [6, 7, 8], :]],
                axis=-2)
        return orientations_with_dummy

    def remove_dummy_segments(self, segments):
        nodummy_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        if len(np.shape(segments)) == 1:
            return segments[nodummy_idx]
        else:
            return segments[..., nodummy_idx, :]
