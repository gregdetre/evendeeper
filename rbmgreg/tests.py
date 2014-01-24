from ipdb import set_trace as pause
from nose.tools import set_trace as pause, ok_, eq_
import numpy as np
from numpy.testing import assert_array_equal as arr_eq_
import unittest

from rbm import RbmNetwork, create_random_patternset


class BaseTestCase(unittest.TestCase):
    pass


class RbmTests(BaseTestCase):
    def setUp(self):
        self.net = RbmNetwork(3, 2, 0.01, 0.001)
        self.pset = create_random_patternset(shape=(1,3), npatterns=4)

    def test_update_weights_vectorized(self):
        # confirm that the vectorized version gives the same
        # results as the forloop version
        v_plus = self.pset.get(0)
        d_w1, d_a1, d_b1 = self.net.update_weights_forloop(v_plus)
        d_w2, d_a2, d_b2 = self.net.update_weights_vectorized(v_plus)
        arr_eq_(d_w1, d_w2)
        arr_eq_(d_a1, d_a2)
        arr_eq_(d_b1, d_b2)

if __name__ == '__main__':
    unittest.main()
