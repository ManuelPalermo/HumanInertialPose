
from setuptools import setup
from os import path

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
        "test":  ["pytest", "pytest-flake8", "pytest-cov", "pytest-clarity", "pytest-randomly", "pytest-pydocstyle"],
        "all":   ["pandas", "openpyxl", "matplotlib", "seaborn", "PyQt5", "PyOpenGL", "pyqtgraph", "sphinx", "sphinx-rtd-theme", "sphinx-autoapi", "pytest", "pytest-flake8", "pytest-cov", "pytest-clarity", "pytest-randomly", "pytest-pydocstyle"]
}

setup(name="hipose",
      version=__version__,
      description="Human whole-body pose estimation using inertial sensor data.",
      author="ManuelPalermo",
      author_email="macpalermo@gmail.com",
      packages=["hipose"],
      install_requires=requires_list,
      extras_require=extras,
      url="https://github.com/ManuelPalermo/HumanInertialPose",
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent"],
      )
