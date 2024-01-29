import unittest
import pandas as pd
import numpy as np
import sys

from seq_util import *

class TestNoscFiles(unittest.TestCase):
    
    def setUp(self):
        # Read in the file with all the sequences and NOSC values
        self.long_nosc_df = pd.read_csv('data/genomes/all_ref_prot_NOSC.csv')

    def test_nosc_values(self):
        for idx, row in self.long_nosc_df.iterrows():
            # Check that the NOSC values are as expected. 
            # Running the version that returns NaN rather than raising an error.
            Ce, NC = calc_protein_nosc_no_raise(row.aa_seq)
            if np.isnan(row.NOSC):
                self.assertTrue(np.isnan(Ce/NC))
            else:
                self.assertAlmostEqual(Ce/NC, row.NOSC)

if __name__ == '__main__':
    unittest.main()