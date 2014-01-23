from ipdb import set_trace as pause
from nose.tools import set_trace as pause, ok_, eq_
import numpy as np
import unittest

from rbm import RbmNetwork


class BaseTestCase(unittest.TestCase):
    pass


class RbmTests(BaseTestCase):
    def setUp(self):
        self.net = RbmNetwork((2,6), (1,6), 0.1)

    def test_init_weights(self):
        # check weight initialization kept them small
        ok_(max(self.net.w.ravel()) < 0.01)

if __name__ == '__main__':
    unittest.main()
