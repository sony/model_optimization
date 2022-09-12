# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

class Cut:
    def __init__(self, op_order, op_record, mem_elements):
        self.op_order = op_order
        self.op_record = op_record
        self.mem_elements = mem_elements

    def memory_size(self):
        # TODO: this is the equivalent to the scheduler's Cut "weight" method,
        #  which needs to compute the Cut's memory size based on the memory elements
        raise NotImplementedError()

    def get_record_names(self):
        return {op.name for op in self.op_record}

    def __eq__(self, other):
        if isinstance(other, Cut):
            # TODO: take care of lists/sets equality here
            return self.op_order == other.op_order and self.op_record == other.op_record and self.mem_elements == other.mem_elements
        return False

    def __hash__(self):
        return hash((frozenset(self.op_order), frozenset(self.op_record), frozenset(self.mem_elements)))