# -*- coding:utf-8 -*-
"""Setuptools of castle."""
import setuptools
import sys

from castle import __version__


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


with open("README.md", "r", encoding='utf-8') as fh:
    long_desc = fh.read()


setuptools.setup(
    name="gcastle",
    version=__version__,
    include_package_data=True,
    python_requires=">=3.6",
    author="Huawei Noah's Ark Lab",
    author_email="zhangkeli1@huawei.com",
    description="gCastle is the fundamental package for causal structure learning with Python.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle",
    packages=setuptools.find_packages('.', exclude=['web']),
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tqdm>=4.48.2",
        "numpy>=1.19.1",
        "pandas>=0.22.0",
        "scipy>=1.7.3",
        "scikit-learn>=0.21.1",
        "matplotlib>=2.1.2",
        "networkx>=2.5",
    ],
)
