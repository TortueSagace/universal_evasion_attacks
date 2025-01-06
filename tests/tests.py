import unittest
import pandas as pd

from fast_carrot import Datanalyser

class ImageTestCase(unittest.TestCase):

    def test_housing(self):
        file = pd.read_csv("../data/housing_train.csv", index_col=0)

        an = Datanalyser(file, ["SalePrice"])

        an.get_characteristics()
        an.choose_best_model()


if __name__ == '__main__':
    unittest.main()