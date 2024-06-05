#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

# TODO: refactor this for a singleton after talk with ofir.

import logging

# Create and configure the logger
logger = logging.getLogger('xquant')
logger.setLevel(logging.INFO)

# Create a file handler
# file_handler = logging.FileHandler('xquant_log_file.log')
# file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handler to the logger
# logger.addHandler(file_handler)
logger.addHandler(console_handler)

LOGGER = logger
