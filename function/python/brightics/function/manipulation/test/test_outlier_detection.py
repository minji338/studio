import unittest
from brightics.common.datasets import load_iris
from brightics.function.transform import split_data
from brightics.function.manipulation import outlier_detection_tukey_carling, outlier_detection_tukey_carling_model, outlier_detection_lof, outlier_detection_lof_model
from sklearn.neighbors import LocalOutlierFactor

class TestOutlierDetection(unittest.TestCase):

    def setUp(self):
        print("*** Outlier Detection UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Outlier Detection UnitTest End ***")

    def test_outlier_detection_tukey_carling(self):
        input_dataframe = self.iris
        
        df_splitted = split_data(table=input_dataframe, train_ratio=7.0, test_ratio=3.0, random_state=1)
        train_df = df_splitted['train_table']
        test_df = df_splitted['test_table']
        
        res_train = outlier_detection_tukey_carling(table=train_df,
                                              input_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        res_model = outlier_detection_tukey_carling_model(table=test_df, model=res_train['model'])
        
        print(res_model['out_table'])
        
        table = res_model['out_table'].values.tolist()
        self.assertListEqual(table[26], [5.5, 4.2, 1.4, 0.2, 'setosa', 'in', 'out', 'in', 'in'])
        self.assertListEqual(table[27], [5.1, 3.8, 1.5, 0.3, 'setosa', 'in', 'in', 'in', 'in'])
        self.assertListEqual(table[28], [6.1, 2.8, 4.7, 1.2 , 'versicolor', 'in', 'in', 'in', 'in'])
        self.assertListEqual(table[29], [6.3, 2.5, 5, 1.9, 'virginica', 'in', 'in', 'in', 'in'])
        self.assertListEqual(table[30], [6.1, 3, 4.6, 1.4, 'versicolor', 'in', 'in', 'in', 'in'])
        
    def test_outlier_detection_lof(self):
        input_dataframe = self.iris
        
        df_splitted = split_data(table=input_dataframe, train_ratio=7.0, test_ratio=3.0, random_state=1)
        train_df = df_splitted['train_table']
        test_df = df_splitted['test_table']
        
        res_train = outlier_detection_lof(table=train_df,input_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        res_model = outlier_detection_lof_model(table=test_df, model=res_train['model'])
        
        print(res_model['out_table'])
        
        table = res_model['out_table'].values.tolist()
        self.assertListEqual(table[0], [5.8, 4, 1.2, 0.2, 'setosa', 'out'])
        self.assertListEqual(table[1], [5.1, 2.5, 3, 1.1, 'versicolor', 'out'])
        self.assertListEqual(table[2], [6.6, 3, 4.4, 1.4 , 'versicolor', 'in'])
        self.assertListEqual(table[3], [5.4, 3.9, 1.3, 0.4, 'setosa', 'in'])
        self.assertListEqual(table[4], [7.9, 3.8, 6.4, 2, 'virginica', 'out'])


if __name__ == '__main__':
    unittest.main()
