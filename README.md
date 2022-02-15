# Human Inertial Pose

![GitHub Workflow Status](https://github.com/ManuelPalermo/HumanInertialPose/actions/workflows/run_tests.yml/badge.svg?branch=main)
![Codecov](https://codecov.io/gh/ManuelPalermo/HumanInertialPose/branch/main/graph/badge.svg)
![PyPI - License](https://img.shields.io/pypi/l/hipose)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hipose)
![PyPI - Version](https://img.shields.io/pypi/v/hipose)

Human whole-body pose estimation using inertial sensor data.



https://user-images.githubusercontent.com/794111/147855142-25f2cc08-d4f8-4aea-9510-814c9f388e3c.mp4


(Demo video: (Left) Xsens-Analyse GT; (Right) MTwAwinda MARG sensor data calibrated and fused using this library.)



# Contains code to
* Calculate sensors orientation (based on [ahrs](https://github.com/Mayitzin/ahrs/)) library.
* Perform sensor-to-segment calibration.
* Perform imu sensor calibration.
* Compute evaluation metrics.
* Define and compute skeleton kinematics (get joint 3D positions and 3D visualization).
* Mapping between some skeleton formats (Xsens(23) / MTwAwinda(17) / XsensUpper(15) / MTwAwindaUpper(11) Ergowear(9))
* Handle some common external exported data (Xsens-Analyse, Xsens-MtManager).
* Parse popular inertial pose datasets (WIP).



---



# Usage
```python
from hipose.data.trial_parsing.extract_xsens_mtmanager import extract_mtmanager_raw_data
example_data_path = "./resources/sample_trials/xsens2mtmanager/example_calibration_xsens/"
imu_data = extract_mtmanager_raw_data(example_data_path)

# initialize filter fusion (example trial has 17 IMUs)
from hipose.api.fusion_filter import InertialPoseFusionFilter
ffilts = InertialPoseFusionFilter(
        num_imus=17, 
        ignore_mag=False,
        fusion_filter_alg="madgwick",
        s2s_calib_method="static_mag",
        default_data_freq=imu_data["freq"],
)

# initialize calibration params from static NPose
# (example trial has 5s of NPose at the start)
calib_s = int(imu_data["freq"] * 5)
ffilts.compute_imus_calibration(acc_calib_data=imu_data["acc"][0:calib_s],
                                gyr_calib_data=imu_data["gyr"][0:calib_s],
                                mag_calib_data=imu_data["mag"][0:calib_s])

# perform filter fusion on trial data to obtain segment orientations
for idx, (acc, gyr, mag) in enumerate(zip(imu_data["acc"][calib_s:],
                                          imu_data["gyr"][calib_s:],
                                          imu_data["mag"][calib_s:])):
    pred_ori = ffilts.update(acc=acc, gyr=gyr, mag=mag)
```

Look at ```/examples/example_visualize_evaluate_inertial_pose.py``` for a more
detailed example showing additional features (e.g. 3D skeleton display,
skeleton mapping, metrics calculation, etc...).



---



# Installation

Minimal Installation
```bash
pip install hipose                                # install package with base requirements
```

Complete Installation
```bash
pip install hipose"[parse,plot,vis3d,test,doc]"   # install package with desired extra dependencies
# pip install hipose"[all]"                       # or install package with all extra dependencies
```


# TODOs

(Pull Request are welcome!)
- [ ] Add parsing utils for commonly used inertial pose estimation datasets in the literature.
  - [ ] [DIP_IMU](https://dip.is.tuebingen.mpg.de/).
  - [ ] [TotalCapture](https://cvssp.org/data/totalcapture/).
- [ ] Improve dynamic optimization s2s calibration method.
- [ ] Rewrite skeleton implementation + improve functionality.
- [ ] Improve unittest coverage.
- [ ] Improve Docs.

---


# Development

### Install latest repository version
```bash
git clone https://github.com/ManuelPalermo/HumanInertialPose.git
cd HumanInertialPose/
pip install -e .                                                        # install package with base requirements
# pip install -e ."[parse,plot,vis3d,test,doc]"  # or ."[all]"          # install package with extra dependencies
```

### Running Unittests

Unittests should ideally be added to cover all implemented code and should be run before submitting any commit!
Running them requires additional dependencies which can be installed with: ```pip install -e ."[test]"```

Tests can be run by calling (from the project root) one of:
```bash
sh test/test.sh               # run all unittests
# sh test/test_complete.sh    # run all unittests + code quality + docs quality
```

### Building docs
Building the api documentation requires additional dependencies which can be installed with: ```pip install -e ."[doc]"```

The docs can be compiled by calling (from the project root):
```bash
cd docs
make html
```

### Common Problems / Fixes

#### 3D Visualizer
* "libGL error: MESA-LOADER: failed to open swrast : .../lib/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /usr/lib/dri/swrast_dri.so) (search paths /usr/lib/dri)":
https://unix.stackexchange.com/questions/655495/trying-to-run-pygame-on-my-conda-environment-on-my-fresh-manjaro-install-and-ge


---


# Acknowledgements

This project was developed from a research scholarship at the 
[BirdLab-UMinho](http://birdlab.dei.uminho.pt/), supported by grant POCI-01-0247-FEDER-39479.
You can check out other works being developed there through the lab's
([Github](https://github.com/BiRDLab-UMinho) / [Instagram](https://www.instagram.com/bird.uminho) / [Facebook](https://www.facebook.com/BiRDLab.Uminho/)). 

A special thanks to the people in the [Ergoaware project](http://birdlab.dei.uminho.pt/ergoaware/) 
who also contributed to the library through helpful discussions, data acquisition and testing.


  
# Citation
If you find the project helpful, please consider citing us (temporary citations):
```
@misc{palermo2022complete,
    author      = {Manuel Palermo and Sara Cerqueira and João André and António Pereira and Cristina P. Santos},
    title       = {Complete Inertial Pose Dataset: from raw measurements to pose with low-cost and high-end MARG sensors},
    year        = {2022},
    eprint      = {2202.06164},
    eprinttype  = {arXiv},
    url         = {https://arxiv.org/abs/2202.06164}
}
```
```
@dataset{palermo2022cipdatabase,
    author       = {Manuel Palermo and Sara Cerqueira and João André and António Pereira and Cristina P. Santos},
    title        = {Complete Inertial Pose (CIP) Dataset},
    year         = {2022},
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.5801928},
    url          = {https://doi.org/10.5281/zenodo.5801928}
}
```
