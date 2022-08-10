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

import operator
from typing import Any, Callable, Dict


class Filter:
    """
    Filter a layer configuration by its attributes.
    """

    def match(self, layer_config: Dict[str, Any]):
        """
        Check whether the passed configuration matches the filter.
        Args:
            layer_config: Layer's configuration to check.

        Returns:
            Whether the passed configuration matches the filter or not.
        """
        raise Exception('Filter did not implement match')


class AttributeFilter(Filter):
    """
    Wrap a key, value and an operation to filter a layer's configuration according to.
    If the layer's configuration has the key, and its' value matches when applying the operator,
    the configuration matches the AttributeFilter.
    """

    def __init__(self,
                 attr: str,
                 value: Any,
                 op: Callable):
        """

        Args:
            attr (str): Attribute to filter a layer's configuration according to.
            value (Any): Value to filter to filter a layer's configuration according to.
            op (Callable): Operator to check if when applied on a layer's configuration value it holds with regard to the filter's value field.
        """
        self.attr = attr
        self.value = value
        self.op = op

    def __eq__(self, other: Any) -> bool:
        """
        Check whether an object is equal to the AttributeFilter or not.

        Args:
            other: Object to check if it is equal to the AttributeFilter or not.

        Returns:
            Whether the object is equal to the AttributeFilter or not.
        """

        if not isinstance(other, AttributeFilter):
            return False
        return self.attr == other.attr and \
               self.value == other.value and \
               self.op == other.op

    def __or__(self, other: Any):
        """
        Create a filter that combines multiple AttributeFilters with a logic OR between them.

        Args:
            other: Filter to add to self with logic OR.

        Returns:
            OrAttributeFilter that filters with OR between the current AttributeFilter and the passed AttributeFilter.
        """

        if not isinstance(other, AttributeFilter):
            raise Exception("Not an attribute filter. Can not run an OR operation.")
        return OrAttributeFilter(self, other)

    def __and__(self, other: Any):
        """
        Create a filter that combines multiple AttributeFilters with a logic AND between them.

        Args:
            other: Filter to add to self with logic AND.

        Returns:
            AndAttributeFilter that filters with AND between the current AttributeFilter and the passed AttributeFilter.
        """
        if not isinstance(other, AttributeFilter):
            raise Exception("Not an attribute filter. Can not run an AND operation.")
        return AndAttributeFilter(self, other)

    def match(self,
              layer_config: Dict[str, Any]) -> bool:
        """
        Check whether the passed configuration matches the filter.

        Args:
            layer_config: Layer's configuration to check.

        Returns:
            Whether the passed configuration matches the filter or not.
        """
        if self.attr in layer_config:
            return self.op(layer_config.get(self.attr), self.value)
        return False

    def op_as_str(self):
        """

        Returns: A string representation for the filter.

        """
        raise Exception("Filter must implement op_as_str ")

    def __repr__(self):
        return f'{self.attr} {self.op_as_str()} {self.value}'


class OrAttributeFilter(Filter):
    """
    AttributeFilter to filter by multiple filters with logic OR between them.
    """

    def __init__(self, *filters: AttributeFilter):
        """
        Args:
            *filters: List of filters to apply a logic OR between them when filtering.
        """
        self.filters = filters

    def match(self,
              layer_config: Dict[str, Any]) -> bool:
        """
        Check whether a layer's configuration matches the filter or not.

        Args:
            layer_config: Layer's configuration to check.

        Returns:
            Whether a layer's configuration matches the filter or not.
        """

        for f in self.filters:
            if f.match(layer_config):
                return True
        return False

    def __repr__(self):
        """

        Returns: A string representation for the filter.

        """
        return ' | '.join([str(f) for f in self.filters])


class AndAttributeFilter(Filter):
    """
    AttributeFilter to filter by multiple filters with logic AND between them.
    """

    def __init__(self, *filters):
        self.filters = filters

    def match(self,
              layer_config: Dict[str, Any]) -> bool:
        """
        Check whether the passed configuration matches the filter.
        Args:
            layer_config: Layer's configuration to check.

        Returns:
            Whether the passed configuration matches the filter or not.
        """
        for f in self.filters:
            if not f.match(layer_config):
                return False
        return True

    def __repr__(self):
        """

        Returns: A string representation for the filter.

        """
        return ' & '.join([str(f) for f in self.filters])


class Greater(AttributeFilter):
    """
    Filter configurations such that it matches configurations
    that have an attribute with a value that is greater than the value that Greater holds.
    """

    def __init__(self,
                 attr: str,
                 value: Any):
        super().__init__(attr=attr, value=value, op=operator.gt)

    def op_as_str(self): return ">"


class GreaterEq(AttributeFilter):
    """
    Filter configurations such that it matches configurations
    that have an attribute with a value that is greater or equal than the value that GreaterEq holds.
    """

    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.ge)

    def op_as_str(self): return ">="


class Smaller(AttributeFilter):
    """
    Filter configurations such that it matches configurations that have an attribute with a value that is smaller than the value that Smaller holds.
    """

    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.lt)

    def op_as_str(self): return "<"


class SmallerEq(AttributeFilter):
    """
    Filter configurations such that it matches configurations that have an attribute with a value that is smaller or equal than the value that SmallerEq holds.
    """

    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.le)

    def op_as_str(self): return "<="


class NotEq(AttributeFilter):
    """
    Filter configurations such that it matches configurations that have an attribute with a value that is not equal to the value that NotEq holds.
    """

    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.ne)

    def op_as_str(self): return "!="


class Eq(AttributeFilter):
    """
    Filter configurations such that it matches configurations that have an attribute with a value that equals to the value that Eq holds.
    """

    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.eq)

    def op_as_str(self): return "="
