import unittest
import pandas as pd
from brightics.common.datasets import load_iris
from brightics.function.statistics.anova import bartletts_test, oneway_anova


class TestAnova(unittest.TestCase):

    def setUp(self):
        print("*** ANOVA UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** ANOVA UnitTest End ***")

    def test_bartletts_test(self):
        input_dataframe = self.iris

        res = bartletts_test(table=input_dataframe, response_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], factor_col='species')['result']['result_table']
   
        print(res)
        
        table = res.values.tolist()
        self.assertListEqual(table[0], ['sepal_length by species', 16.005701874401502, 0.0003345076070163035])
        self.assertListEqual(table[1], ['sepal_width by species', 2.0910752014392338, 0.35150280041580323])
        self.assertListEqual(table[2], ['petal_length by species', 55.42250284023702, 9.229037733034152e-13])
        self.assertListEqual(table[3], ['petal_width by species', 39.2131139455632, 3.0547839321996904e-09])
        
    def test_oneway_anova(self):
        input_dataframe = self.iris

        res = oneway_anova(table=input_dataframe, response_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], factor_col='species')['result']['result0']
        res = pd.DataFrame([res])
        print(res)
        
        table = res.values.tolist()
        self.assertListEqual(table[0], ["|df|sum_sq|mean_sq|F|PR(>F)|\n|--:|--:|--:|--:|--:|\n|2.0|63.21213333333266|31.60606666666633|119.26450218450334|1.6696691907702582e-31|\n|147.0|38.95620000000001|0.2650081632653062|nan|nan|"])


if __name__ == '__main__':
    unittest.main()
