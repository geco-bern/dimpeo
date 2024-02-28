#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dimpeo",
    version="0.0.1",
    description="",
    author="brdav",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="",
    install_requires=["pytorch-lightning", "torchgeo"],
    packages=find_packages(),
)
