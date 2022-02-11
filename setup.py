
from os import path
import itertools

from setuptools import setup

from hipose import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        requires_list.append(str(line))

extras = {
        "parse": ["pandas", "openpyxl"],
        "plot":  ["matplotlib", "seaborn"],
        "vis3d": ["PyQt5", "PyOpenGL", "pyqtgraph"],
        "doc":   ["sphinx", "sphinx-rtd-theme", "sphinx-autoapi"],
        "test":  ["pytest", "pytest-flake8", "pytest-cov", "pytest-clarity", "pytest-randomly", "pytest-pydocstyle"]
}
extras["all"] = list(itertools.chain(*extras.values()))

setup(name="hipose",
      version=__version__,
      description="Human whole-body pose estimation using MARG multi-sensor data.",
      author="ManuelPalermo",
      author_email="macpalermo@gmail.com",
      url="https://github.com/ManuelPalermo/HumanInertialPose",
      license='MIT',
      packages=["hipose"],
      install_requires=requires_list,
      extras_require=extras,
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent"],
      )
