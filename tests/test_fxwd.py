import unittest
from fxwd import fxwd2rec
import numpy as np

class Test_fxwd(unittest.TestCase):
    def test_samp(self):
        file = './planets.dat'
        colist = [ [0,1], [2,9] ]
        rec  = fxwd2rec(file,colist,[('order',int),('name','|S10')],empstr='---')
        rec_expected = \
            np.array([(1, 'Mercury'), (2, 'Venus  '), (3, 'Earth  '), 
                      (4, 'Mars   '),(9, 'None')],
                     dtype=[('order', int), ('name', '|S10')])

        self.assertTrue( (rec==rec_expected).all() )


if __name__ == '__main__':
    unittest.main()
