
from .base_skeleton import Skeleton

from .xsens import SkeletonXsens, SkeletonXsensUpper
from .ergowear import SkeletonErgowear
from .mtwawinda import SkeletonMTwAwinda

from .mappings import map_segs_xsens2xsensupper, map_kpts_xsens2ergowear, \
    map_segs_xsens2ergowear, map_segs_xsens2mtwawinda, map_segs_mtwawinda2xsens, \
    map_segs_mtwawinda2ergowear, map_segs_ergowear2xsensupper,\
    map_segs_ergowear2mtwawindaupper
