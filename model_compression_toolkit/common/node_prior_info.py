from model_compression_toolkit.common.collectors.statistics_collector import is_number


class NodePriorInfo:
    """
    Class to wrap all prior information we have on a node.
    """

    def __init__(self,
                 min_output: float = None,
                 max_output: float = None,
                 mean_output: float = None,
                 std_output: float = None):
        """
        Initialize a NodePriorInfo object.

        Args:
            min_output: Minimal output value of the node.
            max_output: Maximal output value of the node.
            mean_output: Mean output value of the node.
            std_output: Std output value of the node.
        """

        self.min_output = min_output
        self.max_output = max_output
        self.mean_output = mean_output
        self.std_output = std_output

    def is_output_bounded(self) -> bool:
        """

        Returns: Whether the node's output is bounded within a known range or not.

        """
        return is_number(self.min_output) and is_number(self.max_output)
