"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)

setup(
    name='slim',
    version='0.1',
    package_dir={'': 'slim'},
    include_package_data=True,
    packages=find_packages('slim'),
    description='tf-slim',
)


setup(
    name='dltools',
    version='0.1',
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    package_dir={'': 'hnu_dl_tools'},
    packages=[p for p in find_packages('hnu_dl_tools') if p.startswith('dltools')],
    description='deep learning toolbox',
    url="https://gitee.com/study-cooperation",
    author="HNU deep learning project group",
    platforms="any",
)
