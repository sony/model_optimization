# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import sys
from setuptools import setup, find_packages


def read_install_requires():
    print("Reading install requirments")
    return [r.split('\n')[0] for r in open('requirements.txt', 'r').readlines()]


def get_log_description():
    print("Reading READEME File")
    with open("README.MD", "r") as fh:
        long_description = fh.read()
    return long_description


def get_release_arguments():
    argparser = argparse.ArgumentParser(add_help=False)
    args, unknown = argparser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


args = get_release_arguments()
setup_obj = setup(name='model_compression_toolkit',
                  long_description=get_log_description(),
                  long_description_content_type="text/markdown",
                  description='A Model Compression Toolkit for neural networks',
                  packages=find_packages(
                      exclude=["tests", "tests.*",
                               "requirements", "requirements.*",
                               "tutorials", "tutorials.*"]),
                  classifiers=[
                      "Programming Language :: Python :: 3",
                      "License :: OSI Approved :: Apache Software License",
                      "Operating System :: OS Independent",
                      "Topic :: Scientific/Engineering :: Artificial Intelligence"
                  ],
                  install_requires=read_install_requires(),
                  python_requires='>=3.6'
                  )
