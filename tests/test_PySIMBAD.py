import unittest
from PySIMBAD import names2sim,res2id
import numpy as np

class Test_PySIMBAD(unittest.TestCase):
    def test_names2sim(self):

        names = np.array(['14412','4915'])
        ans = names2sim(names,cat='HD')

        ans_exp = \
            np.array(['result oid\n', 'echodata -n x0\nquery id HD 14412\n',
                   'echodata -n x1\nquery id HD 4915\n'],dtype='|S50')

        self.assertTrue( (ans==ans_exp).all() )

    def test_res2id(self):
        file = './samp-simbad-output.sim'
        idx,oid = res2id(file)
        idx_exp,oid_exp = np.array([ 0.,  1.,  2.,  3.]), \
            np.array([ 1356191.,  1225912.,   759900.,   936604.])

        self.assertTrue( (idx_exp==idx).all() )
        self.assertTrue( (oid_exp==oid).all() )

if __name__ == '__main__':
    unittest.main()
