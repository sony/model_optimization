import time
import unittest


class TimerTestCase(unittest.TestCase):

    times_dict = {}

    @staticmethod
    def add_time_to_dict(test_id, runtime):
        TimerTestCase.times_dict[test_id] = runtime

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('Test runtime (sec): %s: %.3f' % (self.id(), t))
        TimerTestCase.add_time_to_dict(self.id(), t)
