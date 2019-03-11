import unittest
from brightics.common.datasets import load_iris
from brightics.function.statistics import correlation


class TestCorrelation(unittest.TestCase):

    def setUp(self):
        print("*** Correlation UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Correlation UnitTest End ***")

    def test_correlation(self):
        input_dataframe = self.iris

        res = correlation(table=input_dataframe, vars=['sepal_length', 'sepal_width'], method='pearson', height=2.5, corr_prec=2)['result']['corr_table']
   
        print(res)
        
        table = res.values.tolist()
        self.assertListEqual(table[0], ['sepal_width', 'sepal_length', -0.11756978413300201, 0.15189826071144916])


if __name__ == '__main__':
    unittest.main()
