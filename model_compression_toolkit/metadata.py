# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict
from model_compression_toolkit.constants import MCT_VERSION, TPC_VERSION


def get_versions_dict(tpc) -> Dict:
    """

    Returns: A dictionary with TPC and MCT versions.

    """
    # imported inside to avoid circular import error
    from model_compression_toolkit import __version__ as mct_version
    tpc_version = f'{tpc.name}.{tpc.version}'
    return {MCT_VERSION: mct_version, TPC_VERSION: tpc_version}
