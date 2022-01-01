
from .parse_cip_ergowear_dataset import parse_cip_ergowear_dataset,\
    cip_ergowear_database_iterator, report_cip_ergowear_database_status

from .parse_cip_mtwawinda_dataset import parse_cip_mtwawinda_dataset,\
    cip_mtwawinda_database_iterator, report_cip_mtwawinda_database_status


__doc__ = \
    """
    Save data for each modality:
    Modalities       | Description                                                           | Units/Repr  | Shape
    ----------------------------------------------------------------------------------------------------------------
      -imus_acc:      XYZ Linear acceleration measured by the IMUs' Accelerometer             g             (N x S x 3)     
      -imus_gyr:      XYZ Angular velocity measured by the IMUs' Gyroscope                    rad/s         (N x S x 3)
      -imus_mag:      XYZ Magnetic Field measured by the IMUs' Magnetometer                   mT            (N x S x 3)
      -imus_ori:      Orientation of IMUs' computed by a fusion filter                        quat          (N x S x 4)
      -segs_ori:      Orientation of a body segments in world referential                     quat          (N x S x 4)
      -segs_pos:      3D position of the origin of the segment in world referential           m             (N x S x 3)
      -jnts_ang:      Orientation of a body segments relative to the parent segment           rads(ZXY)     (N x S x 4)
      -root_pos:      3D root Δ position at each timestep (like odom)                         m             (N x 1 x 3)
      -root_yaw:      root Δ heading (z-axis orient) at each timestep (like odom)             rads          (N x 1 x 1)
      -com_pos:       3D position of the center of mass in world referential                  m             (N x 1 x 3)
    ----------------------------------------------------------------------------------------------------------------
     * S = num_segments  |  K = num_keypoint

    Files are saved with the following structure inside the .npz file:
    (Modality) =
        {subjectID:
            {sequenceID:
                {repetitionID:
                    {data -> np.ndarray(Nsamples x Segments x Channels)}
                }
            }
        }
    ---------------------------------------------------------------------------------------------------------------
    """
