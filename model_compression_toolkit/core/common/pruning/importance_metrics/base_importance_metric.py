from typing import List

from abc import abstractmethod, ABC

from model_compression_toolkit.core.common import BaseNode


class BaseImportanceMetric(ABC):
    @abstractmethod
    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]):
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_entry_node_to_simd_score method.')  # pragma: no cover




