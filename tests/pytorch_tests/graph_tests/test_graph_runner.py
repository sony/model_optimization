import operator

import torch
import unittest

from model_compression_toolkit.logger import Logger
from tests.pytorch_tests.graph_tests.test_manual_bit_selection import ManualBitWidthByLayerTypeTest, \
    ManualBitWidthByFunctionalLayerTest
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter, NodeNameFilter


class FunctionTestRunner(unittest.TestCase):

    def test_invalid_bit_width_selection_conv(self):
        with self.assertLogs(Logger.get_logger(), level='CRITICAL') as log:
            with self.assertRaises(Exception) as context:
                ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 7).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 7 is invalid for node Conv2d:conv1.")

    def test_invalid_bit_width_selection_add(self):
        with self.assertLogs(Logger.get_logger(), level='CRITICAL') as log:
            with self.assertRaises(Exception) as context:
                ManualBitWidthByFunctionalLayerTest(self, NodeTypeFilter(operator.add), 3).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 3 is invalid for node add:add.")

    def test_manual_bit_width_selection(self):
        """
        This test checks the manual bit-width selection feature.
        """
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 2).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Linear), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(operator.add), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(operator.add), 2).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(torch.nn.Linear)], [2, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(torch.nn.Linear)], [4, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(operator.add)], [2, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(operator.add), NodeTypeFilter(torch.nn.Conv2d)], [4, 4]).run_test()


if __name__ == '__main__':
    unittest.main()
