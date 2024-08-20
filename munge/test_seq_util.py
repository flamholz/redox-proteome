import unittest

from seq_util import *

cwd = path.dirname(__file__)
AA_DF = pd.read_csv(path.join(cwd, '../data/aa_nosc.csv')).set_index('letter_code')


# Each example has a sequence -> (NOSC, NC)
EXAMPLES = {
    # ALA has 3 C atoms and ZC = 0
    'AAAAAAA': (0, 21),
    # GLN has 5 Cs and ZC = 0.4
    'QQQQQQ': (0.4, 30),
    # 3 of each means 24 C and (3*3*0 + 3*5*0.4)/24 = 0.25 
    'AAAQQQ': (0.25, 24),
    # Order doesn't matter
    'QQQAAA': (0.25, 24),
    'AQAQAQ': (0.25, 24),
    
    # Stop codon is ignored
    'AAAAAAA*': (0, 21),
    'QQQQQQ*': (0.4, 30),
    'AAAQQQ*': (0.25, 24),
    'QQQAAA*': (0.25, 24),
    'AQAQAQ*': (0.25, 24),
    
    # Now try adding a third amino
    # TRP has 11 C and ZC = -0.18
    'WWWWW*': (-0.18, 55),
    # 3 of each means 42 C and (3*3*0 + 3*11*-0.18)/42
    'WWWAAA*': (-0.1414285714, 42),
    # 3 different aminos
    'WWWAAAQQQ*': (0.001052631579, 57),
    # Order doesn't matter
    'AAAWWWQQQ*': (0.001052631579, 57),
    'QQQAAAWWW*': (0.001052631579, 57),
    'AWQAWQAWQ*': (0.001052631579, 57),
    
    # Section of spinach rubisco sequence (uniprotkb P00875)
    # TODO: verify this 
    'MSPQTETKASVEFKAGVKDY': (-0.144123711340, 97)
}

class TestSeqUtil(unittest.TestCase):
    
    def test_calc_protein_nosc(self):
        for seq, (NOSC, NC) in EXAMPLES.items():
            res = calc_protein_nosc(seq)
            res_Ce, res_NC = res
            res_NOSC = res_Ce/res_NC
            self.assertEqual(NC, res[1], msg=seq)
            self.assertAlmostEqual(res_NOSC, NOSC, msg=seq)

    def test_calc_nosc_from_smiles(self):
        for aa, row in AA_DF.iterrows():
            if aa == 'U':
                # Formula doesn't work for selenocysteine
                continue
            smiles = row['canonical_SMILES']
            res = calc_nosc_from_smiles(smiles)
            self.assertAlmostEqual(res, row['NOSC'], msg=aa, places=2)


if __name__ == '__main__':
    unittest.main()