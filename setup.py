from setuptools import find_packages
from setuptools import setup


setup(
    packages=find_packages(include=["tunix_accel", "tunix_accel.*"]),
    py_modules=["sitecustomize"],
)
