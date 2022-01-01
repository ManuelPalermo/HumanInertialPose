
import time
import numpy as np

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtGui
except:
    import warnings
    warnings.warn("Additional dependencies are needed for 3D visualization! "
                  "See the README for instructions on how to install them.")

from hipose.rotations import quat_mult, vec_rotate, convert_euler_to_quat


# colors (red/green/blue/white/yellow)
r, g, b, w, y = (1.0, 0, 0, 1.0), (0, 1.0, 0, 1.0), (0, 0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), (0, 1.0, 1.0, 1.0)


class Visualizer3d(object):
    """
    Creates 3D canvas space for visualizing 3D objects in real time
    using pyqtgraph.

    Args:
        animation_fps (float): How many data frames to display per
            second.
        window_freq (float): Frequency to update the window data.
        use_isb_ref(bool): If referential should be transformed to ISB
            recommendation for joint coordinate systems:
            (Y-axis up, X-axis forward, Z-axis right).
        window_title(str): Title of the window created.

    """

    def __init__(self, animation_fps=60.0, window_freq=60.0, use_isb_ref=False,
                 window_title="Display3d"):
        self.animation_fps = animation_fps
        self.window_freq = window_freq
        self._window_timer = time.perf_counter()
        self._animation_timer = time.perf_counter()
        self._set3d_plot(use_isb_ref, window_title)

    def _set3d_plot(self, use_isb_ref=False, window_title="Display3d"):
        # add graphical window
        self._app = QtGui.QApplication([])
        # create window with 3D plot inside
        self.window = gl.GLViewWidget()
        self.window.show()
        self.window.setWindowTitle(window_title)

        # add floor to window
        gz = gl.GLGridItem()
        self.window.addItem(gz)

        # rearranges XYZ referential to camera referential axis and plots it
        self._axis_rot = convert_euler_to_quat(
                [np.pi / 2, 0., 0.] if use_isb_ref else [0, 0, 0],
                seq="XYZ")
        ref_pts = vec_rotate(0.25 * np.eye(3), self._axis_rot)
        self.window.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], ref_pts[0]]),  # X(Red)
                                              color=(1., 0., 0., .5), width=2))
        self.window.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], ref_pts[1]]),  # Y(Green)
                                              color=(0., 1., 0., .5), width=2))
        self.window.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], ref_pts[2]]),  # Z(Blue)
                                              color=(0., 0., 1., .5), width=2))

        self.points3d_plot = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]),
                                                  color=(0.75, 0.75, 0.75, 0.75), size=1)
        self.window.addItem(self.points3d_plot)
        QtGui.QGuiApplication.processEvents()

    def plot_points(self, points, colors=None, radius=3):
        """
        Plots a list of points in the 3D space.

        Args:
            points(np.ndarray[Nx3]): array of 3D points to display.
            colors(None, np.ndarray[Nx3]): array of colors for each of
                the points. Can also receive a single color for all
                points, or None for default color(white).
            radius(float, np.ndarray[Nx1]): radius for each of the
                points, or a single radius value for all points.

        """
        if time.perf_counter() - self._window_timer >= (1 / self.window_freq):
            self._window_timer = time.perf_counter()
            points = self._axis_rot.apply(points)
            if colors is None:
                self.points3d_plot.setData(pos=points, size=radius)
            else:
                self.points3d_plot.setData(pos=points, color=colors, size=radius)
            QtGui.QGuiApplication.processEvents()

        func_compute_time = (time.perf_counter() - self._animation_timer)
        if func_compute_time < (1 / self.animation_fps):
            time.sleep((1 / self.animation_fps) - func_compute_time)
        else:
            self._animation_timer = time.perf_counter()


class SkeletonVisualizer(Visualizer3d):
    """
    Visualizes one or multiple skeletons in 3D space along with optional
    data.

    Args:
        skeletons (dict[Any, Skeleton]): dictionary containing the
            skeletons to be shown(values), associated with desired
            id(key).
        animation_fps (float): How many data frames to display per
            second.
        window_freq (float): Frequency to update the window data.
        skeletons_alpha_color (float, Iterable[float]): Alpha
            transparency for each of the drawn skeletons.
        use_isb_ref(bool): If referential should be transformed to ISB
            recommendation for joint coordinate systems:
            (Y-axis up, X-axis forward, Z-axis right).
        display_segment_axis(bool): if referential axis should be
            displayed for each segment. Can be turned off for
            faster rendering.
        window_title(str): Title of the window created.

    """
    def __init__(self, skeletons, animation_fps=60.0, window_freq=60.0,
                 skeletons_alpha_color=1.0, use_isb_ref=False,
                 display_segment_axis=True, window_title="SkeletonDisplay"):
        super(SkeletonVisualizer, self).__init__(animation_fps, window_freq,
                                                 use_isb_ref, window_title)
        self.skeletons = skeletons
        self.display_segment_axis = display_segment_axis
        self._skeleton_names = dict()
        self._skeleton_segments = dict()
        self._skeleton_joints = dict()
        self._skeleton_segment_axis = dict()

        self._alpha_colors = np.ones((len(skeletons),)) * np.array(skeletons_alpha_color)
        self._set3d_skeleton_plot()

    def _set3d_skeleton_plot(self):
        # initialize skeleton joints/segments
        for s_i, (sk_id, skeleton) in enumerate(self.skeletons.items()):
            # for pyqtgraph.opengl need to change color to [0-1] range and apply desired skeleton alpha
            sk_seg_colors = ((np.array(skeleton.seg_color, dtype=np.float32
                                       ) / np.max(skeleton.seg_color)
                              ) * np.array([1., 1., 1., self._alpha_colors[s_i]]))
            sk_jnt_colors = ((np.array(skeleton.joint_colors, dtype=np.float32
                                       ) / np.max(skeleton.joint_colors)
                              ) * np.array([1., 1., 1., self._alpha_colors[s_i]]))

            # add skeleton joint markers to window
            skeleton_joints = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]),
                                                   color=sk_jnt_colors, size=4)
            self.window.addItem(skeleton_joints)
            self._skeleton_joints[sk_id] = skeleton_joints

            # create and add skeleton segments to window
            segments = []
            segment_axis = []
            for i in range(skeleton.num_segments):
                # create segment line and add to window
                line_segment = gl.GLLinePlotItem(color=tuple(sk_seg_colors[i]), width=2)
                segments.append(line_segment)
                self.window.addItem(line_segment)

                # create segment referential(XYZ) lines and add to window
                if self.display_segment_axis:
                    axis_x_seg = gl.GLLinePlotItem(color=(1., 0., 0., 0.5), width=2)  # X-axis(R)
                    axis_y_seg = gl.GLLinePlotItem(color=(0., 1., 0., 0.5), width=2)  # Y-axis(G)
                    axis_z_seg = gl.GLLinePlotItem(color=(0., 0., 1., 0.5), width=2)  # Z-axis(B)
                    segment_axis.append([axis_x_seg, axis_y_seg, axis_z_seg])

                    self.window.addItem(axis_x_seg)
                    self.window.addItem(axis_y_seg)
                    self.window.addItem(axis_z_seg)

            # store line items to update later
            self._skeleton_segments[sk_id] = segments
            if self.display_segment_axis:
                self._skeleton_segment_axis[sk_id] = segment_axis

            # set skeleton names
            sk_name = gl.GLTextItem(text=sk_id,
                                    font=QtGui.QFont('Helvetica', 12),
                                    color=np.array([255, 255, 255, 255*self._alpha_colors[s_i]])
                                    )
            self.window.addItem(sk_name)
            self._skeleton_names[sk_id] = sk_name

        QtGui.QGuiApplication.processEvents()

    def show3d(self, skeletons_orient_dict, skeletons_root_pos=None):
        """
        Visualize the human skeleton with the desired configuration
            in the 3D space.

        Args:
            skeletons_orient_dict(dict[str, np.ndarray]): dict for
                each skeleton, specifying orientation of each segment.
            skeletons_root_pos(dict[str, np.ndarray[x3]]): dict for
                each skeleton, specifying root joint position. If None,
                then defaults to origin (0,0,0).

        """
        if time.perf_counter() - self._window_timer >= (1 / self.window_freq):
            self._window_timer = time.perf_counter()

            for sk_id, sk_segs_ori in skeletons_orient_dict.items():
                skeleton = self.skeletons[sk_id]

                sk_root_pos = (None if (skeletons_root_pos is None)
                               else skeletons_root_pos[sk_id])

                # obtain keypoints 3d positions
                sk_pos3d = skeleton.compute_skeleton_kpts_from_seg_orient(
                        sk_segs_ori,
                        root_pos=sk_root_pos
                )

                # plot 3d name above skeleton
                mean_pos = sk_pos3d.mean(axis=0)
                max_height = sk_pos3d.max(axis=0)[2]
                self._skeleton_names[sk_id].setData(pos=[mean_pos[0], mean_pos[1]-0.1, max_height+0.2])

                # plot 3D joint markers
                sk_pos3d = vec_rotate(sk_pos3d, self._axis_rot)
                self._skeleton_joints[sk_id].setData(pos=sk_pos3d)

                # plot segments
                for i in range(skeleton.num_segments):
                    line_points = np.array([sk_pos3d[skeleton.seg_start_pt[i]],
                                            sk_pos3d[skeleton.seg_end_pt[i]]])
                    self._skeleton_segments[sk_id][i].setData(pos=line_points)

                if self.display_segment_axis:

                    ### TODO: remove temporary workaround for skeleton class (dummy segments)!
                    sk_segs_ori = self.skeletons[sk_id].add_dummy_segments(sk_segs_ori)
                    ###
                    sk_rotaxis = quat_mult(self._axis_rot, sk_segs_ori)
                    for i in range(skeleton.num_segments):
                        axis_ref_pts = 0.075 * np.eye(3)
                        r = self._skeleton_segment_axis[sk_id][i]
                        if r is not None:
                            x_axis, y_axis, z_axis = r
                            x_axis.setData(
                                    pos=np.array(
                                            [[0, 0, 0], vec_rotate(axis_ref_pts[0], sk_rotaxis[i])
                                             ]) + sk_pos3d[skeleton.seg_start_pt[i]])
                            y_axis.setData(
                                    pos=np.array(
                                            [[0, 0, 0], vec_rotate(axis_ref_pts[1], sk_rotaxis[i])
                                             ]) + sk_pos3d[skeleton.seg_start_pt[i]])
                            z_axis.setData(
                                    pos=np.array(
                                            [[0, 0, 0], vec_rotate(axis_ref_pts[2], sk_rotaxis[i])
                                             ]) + sk_pos3d[skeleton.seg_start_pt[i]])

            QtGui.QGuiApplication.processEvents()

        # limit frame display speed to desired fps
        func_compute_time = (time.perf_counter() - self._animation_timer)
        if func_compute_time < (1 / self.animation_fps):
            time.sleep((1 / self.animation_fps) - func_compute_time)
        else:
            self._animation_timer = time.perf_counter()
