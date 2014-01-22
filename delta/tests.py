from ipdb import set_trace as pause
from nose.tools import set_trace as pause, ok_, eq_
import unittest


class BaseTestCase(unittest.TestCase):
    pass

class Tests(BaseTestCase):
#     def setUp(self):
#         pass

    pass

if __name__ == '__main__':
    unittest.main()
