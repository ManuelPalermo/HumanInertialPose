import pytest
import warnings

warnings.filterwarnings("ignore")
# flake8: noqa: F401


@pytest.mark.filterwarnings("ignore")
def test_requirement_imports():
    import numpy
    import scipy
    import ahrs


@pytest.mark.filterwarnings("ignore")
def test_pkg_imports():
    import hipose

    import hipose.data.trial_parsing
    import hipose.data.trial_parsing.extract_ergowear
    import hipose.data.trial_parsing.extract_xsens_mtmanager
    import hipose.data.trial_parsing.extract_xsens_analyse

    import hipose.data.dataset_parsing
    import hipose.data.dataset_parsing.parse_cip_ergowear_dataset
    import hipose.data.dataset_parsing.parse_cip_mtwawinda_dataset


    import hipose.imu
    import hipose.rotations
    import hipose.metrics
    import hipose.utils
