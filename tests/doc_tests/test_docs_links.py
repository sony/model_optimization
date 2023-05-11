# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest
import subprocess
from shutil import rmtree
from os import walk, getcwd
from os.path import join, isdir, isfile
import requests
import re


class TestDocsLinks(unittest.TestCase):

    @staticmethod
    def check_link(_url):
        try:
            response = requests.get(_url)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Error checking link '{_url}': {e}")
            return False

    def test_readme_files(self):
        cwd = getcwd()
        print('Current working directory: ', cwd)

        mct_folder = join(cwd, "model_optimization")
        for filepath, _, filenames in walk(mct_folder):
            for filename in filenames:
                if filename.endswith(".md"):
                    with open(join(filepath, filename), "r") as fh:
                        lines = fh.readlines()
                        for i, l in enumerate(lines):
                            _strs = re.findall(r'\[.[^]]*\]\(.[^)]*\)', l)
                            for link_str in _strs:
                                _link = link_str.split(']')[-1][1:-1]
                                if _link[0] == '#':
                                    pass
                                elif 'http://' in _link or 'https://' in _link:
                                    self.assertTrue(self.check_link(_link),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
                                else:
                                    _link = _link.split('#')[0]
                                    self.assertTrue(isdir(join(filepath, _link)) or isfile(join(filepath, _link)),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
            rmtree(mct_folder)
