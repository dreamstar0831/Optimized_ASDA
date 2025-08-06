#!/usr/bin/env python
# coding=utf-8
# This module is a copy of origin version from Jiajia Liu, https://github.com/PyDL/ASDA.
# by Quan Xie@University of Science and Technology of China.
# August 2025

from setuptools import setup, find_packages
import glob
import os


def main():
    setup(
        name="Optimzed ASDA",
        python_requires='>3.5.0',
        version="2.2",
        author="Quan Xie",
        author_email="xq30@mail.ustc.edu.cn",
        description=("Optimized Automated Swirl Detection Algorithm"),
        license="GPLv3",
        keywords="Optimized ASDA",
        url="https://github.com/dreamstar0831/Optimized_ASDA",
        packages=find_packages(where='.', exclude=(), include=('*',)),
        py_modules=get_py_modules(),

        # dependencies
        install_requires=[
            'numpy',
            'scipy',
            'scikit-image',
            'mpi4py',
            'matplotlib',
            'h5py'
        ],

        classifiers=[
            "Development Status :: 1.0 - Release",
            "Topic :: Utilities",
            "License :: OSI Approved :: GNU General Public License (GPL)",
        ],

        zip_safe=False
    )


def get_py_modules():
    py_modules=[]
    for file in glob.glob('*.py'):
        py_modules.append(os.path.splitext(file)[0])

    print(py_modules)
    return py_modules


if __name__ == "__main__":
    main()
