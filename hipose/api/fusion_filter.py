
import numpy as np

from ahrs.filters import Mahony, Madgwick
from ahrs.common.orientation import ecompass, acc2q

from hipose.imu import apply_imu_calibration, imus_static_calibration
from hipose.rotations import quat_mult, convert_euler_to_quat


class InertialPoseFusionFilter(object):
    def __init__(self, num_imus=17, fusion_filter_alg="madgwick",
                 ignore_mag=False, default_data_freq=100.0,
                 s2s_calib_method="static_mag", output_transform_quats=None,
                 magn_reject_thresholds=(0.90, 1.10)):

        assert fusion_filter_alg in ["mahony", "madgwick"]
        assert s2s_calib_method in [None, "static", "static_mag"]

        # define fusion filter
        self.num_imus = num_imus
        self.fusion_filter_alg = fusion_filter_alg
        self.ignore_mag = ignore_mag
        self.data_freq = default_data_freq
        self.s2s_calib_method = s2s_calib_method
        self.output_transform_quats = output_transform_quats
        self.mreject_thrsh = magn_reject_thresholds

        self.calib_params = dict()  # calibration parameters
        self.fusion_filters = None  # fusion filter for each imu
        self.reset_state()

    def reset_state(self):
        self.calib_params = dict()
        # initialize sensor fusion filter for each IMU
        self.fusion_filters = [self.create_fusion_filter(algorithm=self.fusion_filter_alg,
                                                         freq=self.data_freq)
                               for _ in range(self.num_imus)]

    def compute_imus_calibration(self, acc_calib_data, gyr_calib_data,
                                 mag_calib_data=None,
                                 manual_s2s_alignment=None):
        """
        Computes ergowear calibration parameters from raw IMU
        calibration samples.

        Args:
            acc_calib_data(np.ndarray[NxIx3]): raw accelerometer data
                containing n-pose standing calibration samples.
            gyr_calib_data(np.ndarray[NxIx3]): raw gyroscope data
                containing n-pose standing calibration samples.
            mag_calib_data(None|np.ndarray[NxIx3]): raw magnetometer
                data containing n-pose standing calibration samples.
            manual_s2s_alignment(None|np.ndarray[Ix3]): manual
                rotations to apply to sensor referentials.
        """
        assert (self.s2s_calib_method != "static_mag") or (mag_calib_data is not None), \
            "Magnetometer data is needed when s2s_calib_method='static_mag'."

        mag_calib_data = (mag_calib_data
                          if (mag_calib_data is not None)
                          else np.zeros_like(acc_calib_data))

        # extract calibration parameters from static calibration samples
        calib_params = imus_static_calibration(
                acc_calib_data=acc_calib_data.copy(),
                gyr_calib_data=gyr_calib_data.copy(),
                mag_calib_data=mag_calib_data.copy(),
                manual_align_rots=manual_s2s_alignment,
                s2s_calib_method=self.s2s_calib_method)
        self.calib_params = calib_params
        return calib_params

    def create_fusion_filter(self, algorithm, freq, q0=None):
        """
        Creates filter to fuse raw IMU data to obtain orientation.
        """
        if isinstance(algorithm, str):
            algorithm = algorithm.lower()
            if algorithm == "madgwick":
                filt = Madgwick(frequency=freq, q0=q0)
            elif algorithm == "mahony":
                filt = Mahony(frequency=freq, q0=q0)
            else:
                raise NotImplementedError(
                        f"Chosen orientation algorithm not implemented '{algorithm}'! "
                        "Choose one of ['mahony', 'madgwick'], or pass your own filter object."
                )
        else:
            from copy import deepcopy
            filt = deepcopy(algorithm)
        return filt

    def update(self, acc, gyr, mag=None, dt=None):
        """
        Processes sample of inertial data. Applies calibration
        and calculates orientation.

        Args:
            acc(np.ndarray[Ix3]): accelerometer data from each IMU.
            gyr(np.ndarray[Ix3]): gyroscope data from each IMU.
            mag(None|np.ndarray[Ix3]): magnetometer data from each IMU.
            dt(float): time elapsed since last sample.

        Return:
            np.ndarray[Ix4]: segments_orientation

        """
        assert (self.ignore_mag) or (mag is not None), \
            "No magnetometer data as passed and ignore_mag=False."
        mag = mag if (mag is not None) else np.zeros_like(acc)

        assert acc.shape == gyr.shape == mag.shape, \
            f"Not all inertial data has the same dimenions:" \
            f" [{acc.shape}, {gyr.shape}, {mag.shape}]"
        assert acc.shape[0] == self.num_imus

        acc, gyr, mag = apply_imu_calibration(
                acc, gyr, mag,
                acc_magn=self.calib_params["acc_magn"][:, None],
                gyr_bias=self.calib_params["gyr_bias"],
                mag_magn=self.calib_params["mag_magn"][:, None],
                s2s_offset=self.calib_params["s2s_offset"])

        segs_orient = []
        # compute orientation of each imu using fusion-filter (could be vectorized)
        for s, (s_acc, s_gyr, s_mag) in enumerate(zip(acc, gyr, mag)):
            # ignore mag if specified or magnetic disturbances noticed
            mag_norm = np.linalg.norm(s_mag, axis=-1)
            mag_reject = (not (self.mreject_thrsh[0] < mag_norm < self.mreject_thrsh[1]))

            # gyroscope / accelerometer update
            if self.ignore_mag or mag_reject:

                # compute q0 if None
                self.fusion_filters[s].q0 = (
                        acc2q(s_acc)
                        if self.fusion_filters[s].q0 is None
                        else self.fusion_filters[s].q0
                )
                # compute next state
                s_orient = self.fusion_filters[s].updateIMU(
                        self.fusion_filters[s].q0, gyr=s_gyr, acc=s_acc
                )

            # gyroscope / accelerometer / magnetometer update
            else:

                # compute q0 if None
                self.fusion_filters[s].q0 = (
                        ecompass(s_acc, s_mag, representation="quaternion", frame="NED")
                        if self.fusion_filters[s].q0 is None else
                        self.fusion_filters[s].q0
                )

                # compute next state
                s_orient = self.fusion_filters[s].updateMARG(
                        self.fusion_filters[s].q0, gyr=s_gyr, acc=s_acc, mag=s_mag
                )

            # store orientation for next iteration
            self.fusion_filters[s].q0 = s_orient.copy()
            segs_orient.append(s_orient)

        # and rotate orientation from NED (computed by AHRS library) to NWU (used by us)
        segs_orient = np.array(segs_orient)
        segs_orient = quat_mult(convert_euler_to_quat([np.pi, 0, 0], seq="XYZ"), segs_orient)
        segs_orient = segs_orient / np.linalg.norm(segs_orient, axis=-1, keepdims=True)

        if self.output_transform_quats is not None:
            segs_orient = quat_mult(segs_orient, self.output_transform_quats)

        return segs_orient
