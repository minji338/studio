import unittest
from brightics.common.datasets import load_iris
from brightics.function.clustering import hierarchical_clustering, hierarchical_clustering_post


class TestHierarchicalClustering(unittest.TestCase):
    def setUp(self):
        print("*** Hierarchical Clustering UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Hierarchical Clustering UnitTest End ***")

    def test_hierarchical_clustering_post(self):
        input_dataframe = self.iris
        res_clustering = hierarchical_clustering(input_dataframe,
                                              input_cols=['sepal_length', 'sepal_width',
                                                            'petal_length', 'petal_width'])
        res_post_process = hierarchical_clustering_post(res_clustering['model'],
                                                  num_clusters=3)
        
        print(res_post_process['out_table'])
        
        table = res_post_process['out_table'].values.tolist()
        self.assertListEqual(table[49], [5, 3.3, 1.4, 0.2, 'setosa', 'pt_49', 3])
        self.assertListEqual(table[50], [7, 3.2, 4.7, 1.4, 'versicolor', 'pt_50', 1])
        self.assertListEqual(table[51], [6.4, 3.2, 4.5, 1.5, 'versicolor', 'pt_51', 1])
        self.assertListEqual(table[52], [6.9, 3.1, 4.9, 1.5, 'versicolor', 'pt_52', 1])
        self.assertListEqual(table[53], [5.5, 2.3, 4, 1.3, 'versicolor', 'pt_53', 2])


if __name__ == '__main__':
    unittest.main()