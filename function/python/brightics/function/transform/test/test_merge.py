import unittest
from brightics.common.datasets import load_iris
from brightics.function.transform.merge import bind_row_column


class TestMerge(unittest.TestCase):

    def setUp(self):
        print("*** Merge UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Manipulation UnitTest End ***")

    def test_bind_row_column1(self):
        input_dataframe = self.iris
        
        res = bind_row_column(first_table=input_dataframe, second_table=input_dataframe, row_or_col='row')['table']
        
        print(res)
        
        self.assertEqual(300, len(res.index))
        self.assertEqual(5, res.columns.size)
        
    def test_bind_row_column2(self):
        input_dataframe = self.iris

        res = bind_row_column(first_table=input_dataframe, second_table=input_dataframe, row_or_col='col')['table']
        
        print(res)
        
        self.assertEqual(150, len(res.index))
        self.assertEqual(10, res.columns.size)
        
    
if __name__ == '__main__':
    unittest.main()
