# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


"""
Created on April 20, 2018

@author: Hai Habi
"""
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
    argparser.add_argument('--new_version', help='required new version argument for releasing a package', required=True)
    args, unknown = argparser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args


args = get_release_arguments()
setup_obj = setup(name='constrained_model_optimization',
                  version=args.new_version,
                  long_description=get_log_description(),
                  long_description_content_type="text/markdown",
                  description='A Constrained Model Optimization for neural network.',
                  packages=find_packages(
                      exclude=["tests", "tests.*",
                               "requirements", "requirements.*",
                               "model_zoo", "model_zoo.*"
                               "tutorials", "tutorials.*"]),
                  classifiers=[
                      "Programming Language :: Python :: 3",
                      "License :: OSI Approved :: MIT License",
                      "Operating System :: OS Independent",
                  ],
                  install_requires=read_install_requires(),
                  python_requires='>=3.6'
                  )
