from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ss_segmentation'],
    scripts=['nodes/ss_inference'],
    package_dir={'': 'src'}
)

setup(**d)
