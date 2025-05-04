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
import os
import unittest
import subprocess
from shutil import rmtree
from os import walk, getcwd, getenv
from os.path import join, isdir, isfile
from urllib.parse import urlparse

import requests
import re


# def _check_local_file_exists(link_path):
#
#
#     # repo_root = f'model_optimization/blob/{branch_name}/'
#     # abs_path = os.path.join(repo_root, link_path)
#
#     assert os.path.isfile(link_path) or os.path.isdir(link_path), f"Missing file or directory: {link_path}"

def parse_github_blob_url(url):

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    if len(parts) < 5 or parts[2] != "blob":
        raise ValueError("Invalid GitHub blob URL")

    owner, repo = parts[0], parts[1]
    branch = parts[3]
    filepath = "/".join(parts[4:])

    return owner, repo, branch, filepath


class TestDocsLinks(unittest.TestCase):
    """
    A test for checking links in 'readme' (.md files), notebooks (.ipynb files) and '.rst' files.
    Searches for external links (https:) and local folder or files links.
    """

    # @staticmethod
    # def check_link(_url, branch_name):
    #     if 'sony/model_optimization' in _url:
    #         _url = _url.replace('/main/', f'/{branch_name}/')
    #
    #         index = _url.find(f"/{branch_name}/")
    #         if index != -1:
    #             link_path = _url[index + len(f"/{branch_name}/"):]
    #         else:
    #             print(f"The substring '/{branch_name}/' was not found.")
    #             raise Exception()
    #
    #         if '#L' in _url:
    #             parts = link_path.split('#L')
    #             link_path = parts[0]
    #             line_num = int(parts[1]) if len(parts) > 1 else None
    #             _check_local_file_exists(link_path)
    #
    #             with open(link_path, 'r', encoding='utf-8') as f:
    #                 lines = f.readlines()
    #                 if line_num <= len(lines):
    #                     return True, f"Line {line_num} exists in {link_path}"
    #                 else:
    #                     return False, f"Line {line_num} does NOT exist in {link_path} (file has {len(lines)} lines)"
    #
    #         _check_local_file_exists(_url)
    #         return True
    #     else:
    #         try:
    #             response = requests.get(_url)
    #             print(f"[{response.status_code}] {_url}")
    #             if response.status_code == 200:
    #                 return True
    #         except Exception as e:
    #             print(f"Error checking link '{_url}': {e}")
    #
    #     # try:
    #     #     _url = _url.replace('/main/', f'/{branch_name}/')
    #     #     response = requests.get(_url)
    #     #     if response.status_code == 200:
    #     #         return True
    #     # except Exception as e:
    #     #     print(f"Error checking link '{_url}': {e}")
    #
    #     return False

    @staticmethod
    def check_link(_url, branch_name):
        try:
            response = requests.head(_url, allow_redirects=True)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Error checking link '{_url}': {e}")
        try:
            _url = _url.replace('/main/', f'/{branch_name}/')
            response = requests.get(_url)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Error checking link '{_url}': {e}")

        try:
            # GitHub action have limited requests-rate for 'blob' pages
            if 'blob' in _url:
                owner, repo, branch, filepath = parse_github_blob_url(_url)
                _url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}?ref={branch}"
                print(_url)
                response = requests.head(_url, allow_redirects=True)
                print(f'   ===> Response: {response}')
                if response.status_code == 200:
                    return True
        except Exception as e:
            print(f"Error checking link '{_url}': {e}")

        return False

    def test_readme_and_rst_files(self):
        mct_folder = getcwd()
        print("MCT folder:", mct_folder)
        branch_name = getenv("GITHUB_HEAD_REF")
        print("Branch name:", branch_name)
        are_links_ok = True
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
                                if '/model_optimization/blob/main/' in _link:
                                    _link = join(mct_folder, _link.split('/model_optimization/blob/main/')[1])
                                if _link[0] == '#':
                                    # A link starting with '#' is a local reference to a headline in the current file --> ignore
                                    pass
                                elif _link.startswith('data:image/'):
                                    # A link starting with 'data:image/' is a base64-encoded image --> ignore
                                    print("Inline image, skipping.")
                                elif 'http://' in _link or 'https://' in _link:
                                    if self.check_link(_link, branch_name):
                                        print("Link ok:", _link)
                                    else:
                                        are_links_ok = False
                                        print(f'Broken link: {_link} in {join(filepath, filename)}')
                                else:
                                    _link = _link.split('#')[0]
                                    if isdir(join(filepath, _link)) or isfile(join(filepath, _link)):
                                        print("Link ok:", _link)
                                    else:
                                        are_links_ok = False
                                        print(f'Broken link: {_link} in {join(filepath, filename)}')

                elif filename.endswith(".rst"):
                    # doc source file detected. go over lines in search of links.
                    with open(join(filepath, filename), "r") as fh:
                        lines = fh.readlines()
                        for i, l in enumerate(lines):
                            # Search links in doc source files (.rst) with regular expressions:
                            # format: search for a string between <>, which is the link
                            _strs = re.findall(r"<([^<>]+)>", l)
                            for _link in _strs:
                                if '/model_optimization/blob/main/' in _link:
                                    _link = join(mct_folder, _link.split('/model_optimization/blob/main/')[1])
                                if _link.startswith('ug-'):
                                    # A link starting with 'ug-' is a reference to another .rst file --> ignore
                                    # This link is checked when generating the docs
                                    pass
                                elif 'http://' in _link or 'https://' in _link:
                                    if self.check_link(_link, branch_name):
                                        print("Link ok:", _link)
                                    else:
                                        are_links_ok = False
                                        print(f'Broken link: {_link} in {join(filepath, filename)}')
                                else:
                                    if isfile(join(filepath, _link.replace('../', '') + '.rst')):
                                        print("Link ok:", _link)
                                    else:
                                        are_links_ok = False
                                        print(f'Broken link: {_link} in {join(filepath, filename)}')
        self.assertTrue(are_links_ok, msg='Found broken links!')
