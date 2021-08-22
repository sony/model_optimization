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
            folder: Path of folder with images to load.
            preprocessing: List of functions to use when processing the images before retrieving them.
            batch_size: Number of images to retrieve each sample.
            file_types: Files types to scan in the folder. Default list is :data:`~network_optimization_package.common.data_loader.FILETYPES`

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
        print(f"Finished Disk Scanning: Found {self.n_files} files")
        self.preprocessing = preprocessing
        self.batch_size = batch_size

    def _sample(self):
        """
        Read batch_size random images from the image_list the FolderImageLoader holds.
        Process them using the preprocessing list that was passed at initialization, and
        prepare it for retrieving.
        """

        index = np.random.randint(0, self.n_files - 1, self.batch_size)
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
