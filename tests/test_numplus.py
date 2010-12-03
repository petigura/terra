import unittest
from numplus import binavg
import numpy as np

class Test_binavg(unittest.TestCase):
    def test_samp(self):

        x = np.array([1,2,3,4])
        y = np.array([1,3,2,4])
        bins = np.array([0,2.5,5])

        binx,biny = binavg(x,y,bins)
        binx_exp, biny_exp = np.array([1.25,3.75]),np.array([2.,3.])

        self.assertTrue( (binx_exp==binx).all() )
        self.assertTrue( (biny_exp==biny).all() )

if __name__ == '__main__':
    unittest.main()
