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
import os
os.environ['CURL_CA_BUNDLE'] = ''


class TestDocsLinks(unittest.TestCase):
    """
    A test for checking links in 'readme' (.md files), notebooks (.ipynb files) and '.rst' files.
    Searches for external links (https:) and local folder or files links.
    """

    @staticmethod
    def check_link(_url):
        try:
            response = requests.get(_url)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Error checking link '{_url}': {e}")
            return False

    def test_readme_and_rst_files(self):
        mct_folder = getcwd()
        print("MCT folder:", mct_folder)
        for filepath, _, filenames in walk(mct_folder):
            for filename in filenames:

                if filename.endswith(".md") or filename.endswith(".ipynb"):
                    # readme file detected. go over lines in search of links.
                    with open(join(filepath, filename), "r") as fh:
                        lines = fh.readlines()
                        for i, l in enumerate(lines):
                            # Search links in readme files with regular expressions:
                            # format: search for a string between [] followed by a string in (). the latter is the link.
                            _strs = re.findall(r'\[.[^]]*\]\(.[^)]*\)', l)
                            for link_str in _strs:
                                _link = link_str.split(']')[-1][1:-1]
                                # replace colab link with actual github link because accessing a broken link through colab doesn't return an error
                                _link = _link.replace('://colab.research.google.com/github/', '://github.com/')
                                if _link[0] == '#':
                                    # A link starting with '#' is a local reference to a headline in the current file --> ignore
                                    pass
                                elif 'http://' in _link or 'https://' in _link:
                                    self.assertTrue(self.check_link(_link),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
                                    print("Link ok:", _link)
                                else:
                                    _link = _link.split('#')[0]
                                    self.assertTrue(isdir(join(filepath, _link)) or isfile(join(filepath, _link)),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
                                    print("Link ok:", _link)

                elif filename.endswith(".rst"):
                    # doc source file detected. go over lines in search of links.
                    with open(join(filepath, filename), "r") as fh:
                        lines = fh.readlines()
                        for i, l in enumerate(lines):
                            # Search links in doc source files (.rst) with regular expressions:
                            # format: search for a string between <>, which is the link
                            _strs = re.findall(r"<([^<>]+)>", l)
                            for _link in _strs:
                                if _link.startswith('ug-'):
                                    # A link starting with 'ug-' is a reference to another .rst file --> ignore
                                    # This link is checked when generating the docs
                                    pass
                                elif 'http://' in _link or 'https://' in _link:
                                    self.assertTrue(self.check_link(_link),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
                                    print("Link ok:", _link)
                                else:
                                    self.assertTrue(isfile(join(filepath, _link.replace('../', '') + '.rst')),
                                                    msg=f'Broken link: {_link} in {join(filepath, filename)}')
                                    print("Link ok:", _link)
