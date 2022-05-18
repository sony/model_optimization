
from typing import List

from model_compression_toolkit.common.network_editors.edit_network import EditRule


class DebugConfig:
    """
    A class for MCT core debug information.
    """
    def __init__(self, analyze_similarity: bool = False,
                 network_editor: List[EditRule] = []):
        """

        Args:

            analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is
             enabled) or not. Can be used to pinpoint problematic layers in the quantization process.
            network_editor (List[EditRule]): A list of rules and actions to edit the network for quantization.
        """
        self.analyze_similarity = analyze_similarity
        self.network_editor = network_editor
