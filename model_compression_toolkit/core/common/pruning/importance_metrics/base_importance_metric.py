from typing import List

from abc import abstractmethod, ABC

from model_compression_toolkit.core.common import BaseNode


class BaseImportanceMetric(ABC):

    @abstractmethod
    def get_entry_node_to_score(self, sections_input_nodes:List[BaseNode]):
        raise Exception



