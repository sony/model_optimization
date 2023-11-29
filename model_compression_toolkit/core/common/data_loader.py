# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Callable

import numpy as np
from PIL import Image

#:
FILETYPES = ['jpeg', 'jpg', 'bmp', 'png']


class FolderImageLoader(object):
    """

    Class for images loading, processing and retrieving.

    """

    def __init__(self,
                 folder: str,
                 preprocessing: List[Callable],
                 batch_size: int,
                 file_types: List[str] = FILETYPES):

        """ Initialize a FolderImageLoader object.

        Args:
            folder: Path of folder with images to load. The path has to exist, and has to contain at
            least one image.
            preprocessing: List of functions to use when processing the images before retrieving them.
            batch_size: Number of images to retrieve each sample.
            file_types: Files types to scan in the folder. Default list is :data:`~model_compression_toolkit.core.common.data_loader.FILETYPES`

        Examples:

            Instantiate a FolderImageLoader using a directory of images, that returns 10 images randomly each time it is sampled:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[], batch_size=10)
            >>> images = image_data_loader.sample()

            To preprocess the images before retrieving them, a list of preprocessing methods can be passed:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[lambda x: (x-127.5)/127.5], batch_size=10)

            For the FolderImageLoader to scan only specific files extensions, a list of extensions can be passed:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[], batch_size=10, file_types=['png'])

        """

        self.folder = folder
        self.image_list = []
        print(f"Starting Scanning Disk: {self.folder}")
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                file_type = file.split('.')[-1].lower()
                if file_type in file_types:
                    self.image_list.append(os.path.join(root, file))
        self.n_files = len(self.image_list)
        assert self.n_files > 0, f'Folder to load can not be empty.'
        print(f"Finished Disk Scanning: Found {self.n_files} files")
        self.preprocessing = preprocessing
        self.batch_size = batch_size

    def _sample(self):
        """
        Read batch_size random images from the image_list the FolderImageLoader holds.
        Process them using the preprocessing list that was passed at initialization, and
        prepare it for retrieving.
        """

        index = np.random.randint(0, self.n_files, self.batch_size)
        image_list = []
        for i in index:
            file = self.image_list[i]
            img = np.uint8(np.array(Image.open(file).convert('RGB')))
            for p in self.preprocessing:  # preprocess images
                img = p(img)
            image_list.append(img)
        self.next_batch_data = np.stack(image_list, axis=0)

    def sample(self):
        """

        Returns: A sample of batch_size images from the folder the FolderImageLoader scanned.

        """

        self._sample()
        data = self.next_batch_data  # get current data
        return data
