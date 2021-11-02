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


import numpy as np
import os
import shutil
import unittest
from PIL import Image
from pathlib import Path

from model_compression_toolkit import FolderImageLoader

img_path = "./test_data_loader_dir/test_img.jpeg"
img_shape = (224, 224, 3)
sample_batch = 5


class TestLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = os.path.dirname(img_path)
        Path(x).mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(np.random.random(img_shape).astype(np.uint8))
        im.save(img_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.dirname(img_path))

    def test_data_loader(self):
        folder = os.path.dirname(img_path)
        imgs_loader = FolderImageLoader(folder=folder,
                                        preprocessing=[],
                                        batch_size=sample_batch)
        s = imgs_loader.sample()
        self.assertTrue(isinstance(s, np.ndarray))
        self.assertTrue(s.shape == (sample_batch,) + img_shape)

    def test_empty_folder(self):
        os.remove(os.path.abspath(img_path))
        folder = os.path.dirname(img_path)
        with self.assertRaises(Exception):
            FolderImageLoader(folder=folder,
                              preprocessing=[],
                              batch_size=sample_batch)


if __name__ == '__main__':
    unittest.main()
