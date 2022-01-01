
# TODO: Skeleton class (and children classes) need great rework!!
# Can be used for debugging existing skeleton structures for now!

# - the dummy segments bullshit was a hacky way to compute/visualize joints with
#       offsets (for example, in many skeletons e.g. xsens, the pelvis segment
#       origin is on the middle point between the hips, but the leg child-segments
#       origin is on both outer sides of the hips, with a constant connection
#       with relative orientation) (this bullshit took way longer than a principled
#       version and is garbage, but I don't feel like looking at it anymore and it
#       seems to give expected results).
# - classical presentation using DH parameters is not very helpful since
#       IMUs give absolute segment orientations which cannot be used to
#       calculate standard FK / Ik without constant back-forth conversions.
# - params are being stored in multiple separate arrays which is not very
#       organized / intuitive but facilitates some things (although
#       transformation matrices between referential's would be better)
# - work in terms of transformation referential's relative to world frame
#       more intuitive when working with absolute orientations for each segment.

# Options:
# (would also require adapting it to the visualizer)
# - Use of KDL library could facilitate kinematics and referential transforms
#       (also used by ROS), but requires URDF files or similar to instantiate
#       skeletons and adds more dependencies.
#       https://github.com/foolyc/PyKDL

# - adopt this library instead? pretty close to what we want and implemented by FAIR
#       requires some changes to current methods but would be more principled
#       methods to deal with global/local frames
#       https://github.com/facebookresearch/fairmotion/blob/master/fairmotion/core/motion.py


import numpy as np

from hipose.rotations import quat_mult, vec_rotate

# colors (red/green/blue/white/yellow)
r, g, b  = (255, 0, 0, 255),     (0, 255, 0, 255),   (0, 0, 255, 255)      # red / green / blue
w, k, gr = (255, 255, 255, 255), (0, 0, 0, 255),     (127, 127, 127, 200)  # white / black / gray
y, c, p  = (255, 255, 0, 255),   (0, 127, 127, 255), (127, 0, 127, 255)    # yellow / cyan / purple


class Skeleton(object):
    """
    Defines a skeleton structure and properties associated with it. All proprieties are
    converted in order to be in relation to the set of segments/joints selected for use.

    Args:
        root_joint (int): joint which is considered the root for the kinematic chain.
        joint_names (list[str]): list names of each of the joints in the skeleton model.
        segment_starts(list[int]): list of indexes of joints where a segment starts.
        segment_ends(list[int]): list of indexes of joints where a segment ends.
        joint_colors(list, None): list of colors for each of the defined joints. Used for
            visualization. If None, then all joints are set to white.
        segment_colors(list, None): list of colors for each of the defined segments. Used
            for visualization. If None, then all segments are set to white.
        segment_names(list, None): list names of each of the segments in the skeleton model.
            If None, then segments will have no name("").
        segment_lengths(list, None): list of lengths for each of the skeleton segments. If
            None, defaults all lengths to 0.1m.
        segment_angles(list, None): [still under changes] default global angles for each of
            the segments.
        joints_reorder(list, None): new indexes for each of the joints passed, used to
            reorder the joints order (e.g. to match data from multiple datasets9

    """
    def __init__(self, root_joint, joint_names, segment_starts, segment_ends,
                 joint_colors=None, segment_colors=None, segment_names=None,
                 segment_lengths=None, segment_angles=None, joints_reorder=None):

        # get mapping from complete joint set to selected ones (idx[i] = old_idx)
        # joints can also be reordered if requested(for example to use
        # different datasets together)
        ###########################################################################################
        self.kpts_remapping = sorted(list(set(segment_starts + segment_ends)))
        if joints_reorder is not None:
            assert len(joints_reorder) == len(set(joints_reorder)) == len(joint_names), \
                    "Joints reordering idx need to have the same number as selected " \
                    "joints and the idx cannot repeat"
            rel_joints_reorder = [self.kpts_remapping.index(idx) for idx in joints_reorder
                                  if idx in self.kpts_remapping]
            self.kpts_remapping = list(np.array(self.kpts_remapping)[rel_joints_reorder])

        # define segments' start/end, name, color and length
        ###########################################################################################
        self.seg_start_pt = np.array([self.kpts_remapping.index(old_idx)
                                      for old_idx in segment_starts])
        self.seg_end_pt = np.array([self.kpts_remapping.index(old_idx)
                                    for old_idx in segment_ends])

        self.seg_color = (segment_colors if segment_colors is not None
                          else [w for _ in range(self.num_segments)])
        self.seg_names = (segment_names if segment_names is not None
                          else ['' for _ in range(self.num_segments)])
        self.seg_lengths = np.array(segment_lengths if segment_lengths is not None
                                     else [0.1 for _ in range(self.num_segments)])
        self.seg_angles = np.array(segment_angles if segment_angles is not None
                                   else [[1., 0., 0., 0.] for _ in range(self.num_segments)])

        # define joints' root, name, color and parent_hierarchy
        ###########################################################################################
        self.root_joint = self.kpts_remapping.index(root_joint)
        self.joint_names = ([joint_names[j_idx] for j_idx in self.kpts_remapping]
                            if joint_names is not None
                            else ['' for _ in range(self.num_joints)])
        self.joint_colors = ([joint_colors[j_idx] for j_idx in self.kpts_remapping]
                             if joint_colors is not None
                             else [w for _ in range(self.num_joints)])
        self.joint_parents = self._compute_joint_parents(
                self.root_joint, self.seg_start_pt, self.seg_end_pt)

        # keep track of global information and segment order (before selection or reordering)
        ###########################################################################################
        self.joint_names_global = joint_names
        self.seg_start_global_pt = segment_starts
        self.seg_end_global_pt = segment_ends

        assert (len(self.seg_start_pt) == len(self.seg_end_pt)
                == len(self.seg_color) == len(self.seg_names)
                and (self.seg_lengths is None or len(self.seg_lengths) == self.num_segments)), \
            "Num of segment's start_point/end_point/color/names/lengths must be the same!"

        assert ((len(self.joint_names) == len(self.joint_colors))
                and self.root_joint <= self.num_joints), \
            "Num of joint' name/color must be the same and it should be bigger than " \
            "the number of joints being used in segments!"

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def num_segments(self):
        return len(self.seg_start_pt)

    @property
    def has_segment_lengths(self):
        return self.seg_lengths is not None

    def set_segment_lengths(self, seg_lengths):
        """
        Sets the length of each segment. Used as metadata in some operations.
        """
        assert len(self.seg_lengths) == len(seg_lengths)
        self.seg_lengths = seg_lengths

    def compute_segment_vectors(self, segment_orientations, segment_lengths):
        """
        Computes vectors for each segment from global orientation + length.

        Args:
            segment_orientations(np.ndarray[Sx4]): global orientation of
                each segment.
            segment_lengths(np.ndarray[Sx1]): length of each segment.

        Returns:
            (np.ndarray[Sx3]): segment vectors

        """
        # apply rotations to skeleton segment vectors)
        svect = np.c_[segment_lengths, np.zeros((len(segment_lengths), 2))]  # create vector [l, 0, 0]
        return vec_rotate(svect, segment_orientations)

    def compute_segment_vectors_from_keypoints(self, keypoint_pos):
        """
        Computes vectors for each segment from keypoint(start/end)
        positions. Vectors are computed relative to global orientation.
        Also returns segment lengths.

        Args:
            keypoint_positions(np.ndarray[NxKx3]): position of each
                keypoint.

        Returns:
            (np.ndarray[NxSx3], np.ndarray[NxSx1]): segment vectors, segment
                lengths

        """
        # loop can probably be vectorized (add option to process multiple samples 3 dims!)
        if len(keypoint_pos.shape) == 2:
            segment_vectors, segment_lengths = [], []
            for s, e in zip(self.seg_start_pt, self.seg_end_pt):
                svect = keypoint_pos[e] - keypoint_pos[s]
                segment_vectors.append(svect)
                segment_lengths.append(np.linalg.norm(svect))
            segment_vectors = np.array(segment_vectors)
            segment_lengths = np.array(segment_lengths)

        elif len(keypoint_pos.shape) == 3:
            keypoint_pos = keypoint_pos.transpose((1, 0, 2))
            segment_vectors, segment_lengths = [], []
            for s, e in zip(self.seg_start_pt, self.seg_end_pt):
                svect = keypoint_pos[e] - keypoint_pos[s]
                segment_vectors.append(svect)
                segment_lengths.append(np.mean(np.linalg.norm(svect, axis=1)))
            segment_vectors = np.array(segment_vectors).transpose((1, 0, 2))
            segment_lengths = np.array(segment_lengths)

        else:
            raise NotImplementedError

        ### TODO: remove temporary workaround for skeleton class (dummy segments)!
        segment_vectors = self.remove_dummy_segments(segment_vectors)
        segment_lengths = self.remove_dummy_segments(segment_lengths)
        ###

        return segment_vectors, segment_lengths

    def compute_skeleton_kpts_from_seg_orient(self, segment_orients, root_pos=None):
        """
        Compute Keypoint positions from bone vectors using the IMU global
        angles and skeleton structure, following the kinematic chain,
        starting on a root_joint and following the provided connections.

        Args:
            segment_orients(np.ndarray[Sx4]): array of unit
                quaternion with global orientations for each segment.
            skeleton(Skeleton): skeleton object which stores joint
                hierarchy information.
            root_pos(None, Iterable[float]): position of the root joint in
                3D space. If None, defaults to (0,0,0).

        Returns:
            (np.ndarray[Kx4]): 3D positions of each keypoint.

        """
        # rotate referentials so x-axis points in the direction of the next segment
        segment_orients = quat_mult(segment_orients, self.seg_angles)

        ### TODO: remove temporary workaround for skeleton class (dummy segments)!
        segment_orients_glob = self.add_dummy_segments(segment_orients)
        ###

        root_pos = (np.array([0, 0, 0]) if root_pos is None else np.array(root_pos))

        # vect (should have offsets included? - DH params)
        segment_vecs = self.compute_segment_vectors(segment_orients_glob, self.seg_lengths)

        # build kinematic chain offsets
        # (starting on root -> adding children vectors until leaf segments)
        pos3d = {self.root_joint: root_pos}
        for s, (sidx, eidx) in enumerate(zip(self.seg_start_pt, self.seg_end_pt)):
            pos3d[eidx] = pos3d[sidx] + segment_vecs[s]

        # extract 3D positions from dict of transformation matrices (in correct order)
        pos3d = np.array([val for key, val in sorted(pos3d.items())])
        return pos3d

    def _compute_joint_parents(self, root_joint, joint_starts, joint_ends):
        """
        Runs BFS algorithm from root joint to find joint hierarchy.
        """
        def _neighbor_joints(j):
            neighbors = []
            for s, e in zip(joint_starts, joint_ends):
                if s == j: neighbors.append(e)
                elif e == j: neighbors.append(s)
            return neighbors

        found = []
        parents = [-1 for _ in range(self.num_joints)]
        queue = [root_joint]
        while queue:
            joint = queue.pop(0)
            found.append(joint)
            conected_joints = _neighbor_joints(joint)
            for cj in conected_joints:
                if cj not in found:
                    queue.append(cj)
                    parents[cj] = joint
        return parents

    def get_jnts_adj_matrix(self):
        adj = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)
        for j, (s, e) in enumerate(zip(self.seg_start_pt, self.seg_end_pt)):
            adj[s, e] = 1.
            adj[e, s] = 1.
        return adj

    def get_segs_adj_matrix(self):
        if self.__class__.__name__ == "SkeletonErgowear":
            connects = {0: (1,), 1: (0, 2, 3, 6), 2: (1,),
                        3: (1, 4), 4: (3, 5), 5: (4,),
                        6: (1, 7), 7: (6, 8), 8: (7,)}
        elif self.__class__.__name__ == "SkeletonMTwAwinda":
            connects = {0: (1, 11, 14), 1: (0, 2, 3, 7), 2: (1,),
                        3: (1, 4), 4: (3, 5), 5: (4, 6), 6: (5,),
                        7: (1, 8), 8: (7, 9), 9: (8, 10), 10: (9,),
                        11: (0, 12), 12: (11, 13), 13: (12,),
                        14: (0, 15), 15: (14, 16), 16: (15,)}
        else:
            raise NotImplementedError

        adj = np.zeros((len(connects), len(connects)), dtype=np.float32)
        for seg, conn in connects.items():
            for c in conn:
                adj[seg, c] = 1.
                adj[c, seg] = 1.
        return adj

    def __repr__(self):
        """
        Prints available information about the skeleton. This Includes the
        skeleton type, information about joints (idx, name, hierarchy, color),
        information about segments (idx, name, join_start/end, length, color).

        """
        st = ""
        st += "################################# Skeleton Info ####################################"
        st += f"\nSkeleton Type: {self.name} "

        st += f"\n\nJoints({self.num_joints}) Info: \n"
        for j in range(self.num_joints):
            st += f"Joint {j} ".ljust(15)
            st += f"| Name: {self.joint_names[j]} ".ljust(30)
            st += f"| Parent: {self.joint_parents[j]} ".ljust(20)
            st += f"| Color: {self.joint_colors[j]} ".ljust(30)
            st += "\n"

        st += f"\n\nSegments({self.num_segments}) Info: \n"
        for s in range(self.num_segments):
            st += f"Segment {s} ".ljust(15)
            st += f"| Name: {self.seg_names[s]} ".ljust(30)
            st += f"| Start/EndJoint: ({self.seg_start_pt[s]}, {self.seg_end_pt[s]}) ".ljust(30)
            st += f"| Length: {self.seg_lengths[s]} ".ljust(20)
            st += f"| Color: {self.seg_color[s]} ".ljust(30)
            st += "\n"

        st += f"\nAdjacency Matrix Info: \n"
        st += str(self.get_jnts_adj_matrix())
        st += "\n###################################################################################"
        return st

    def add_dummy_segments(self, segments):
        """
        Adds dummy segments to connect joints with no connection. Temporary
        fix because implemented skeleton logic assumes all joints are
        connected hierarchically... Need to change to transformation
        referentials logic were segments are only defined for plotting and
        not needed to calculate the hierarchy.

        Args:
            segments(np.ndarray[Sx4]): orientations for each segment.

        Returns:
            (np.ndarray[(S+D)x4]): orientations_with_dummy.

        """
        raise NotImplementedError

    def remove_dummy_segments(self, segments):
        """
        Reverses the `Skeleton.add_dummy_segments` operation
        """
        raise NotImplementedError
