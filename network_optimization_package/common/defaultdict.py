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


from typing import Callable, Dict, Any


class DefaultDict(object):
    """
    Default dictionary. It wraps a dictionary given at initialization and return its
    values when requested. If the requested key is not presented at initial dictionary,
    it returns the returned value a default factory (that is passed at initialization) generates.
    """

    def __init__(self,
                 known_dict: Dict[Any, Any],
                 default_factory: Callable = None):
        """

        Args:
            known_dict: Dictionary to wrap.
            default_factory: Callable to get default values when requested key is not in known_dict.
        """

        self.default_factory = default_factory
        self.known_dict = known_dict

    def get(self, key: Any) -> Any:
        """
        Get the value of the inner dictionary by the given key, If key is not in dictionary,
        it uses the default_factory to return a default value.

        Args:
            key: Key to use in inner dictionary.

        Returns:
            Value of the inner dictionary by the given key, or a default value if not exist.
            If default_factory was not passed at initialization, it returns None.
        """

        if key in self.known_dict:
            return self.known_dict.get(key)
        else:
            if self.default_factory is not None:
                return self.default_factory()
            return None
