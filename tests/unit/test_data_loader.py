import sys
import unittest
import pandas as pd
sys.path.append('../..')
sys.path.append('.')
from src.controller.data_loader import DataLoader  # noqa: E402


class DataLoaderUnitTest(unittest.TestCase):

    def test(self):
        """
        Test Data Loader
        """
        dl = DataLoader()
        # check multi label binarize
        self.assertRaises(ValueError, dl.multi_label_binarise, pd.Series([]))
        # check multi label binarize
        self.assertRaises(ValueError, dl.tfidf_vectorise, pd.Series([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)