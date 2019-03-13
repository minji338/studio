import unittest
import pandas as pd
from brightics.function.textanalytics import tfidf


class TestTFIDF(unittest.TestCase):

    def setUp(self):
        print("*** TF-IDF UnitTest Start ***")
        data = ['eat turkey on turkey day holiday',
              'i like to eat cake on holiday',
              'turkey trot race on thanksgiving holiday',
              'snail race the turtle',
              'time travel space race',
              'movie on thanksgiving',
              'movie at air and space museum is cool movie',
              'aspiring movie star'
            ]
        df = pd.DataFrame({'text':data})
        print(df)
        self.data = df

    def tearDown(self):
        print("*** TF-IDF UnitTest End ***")

    def test_tfidf1(self):
        input_dataframe = self.data
        
        res = tfidf(table=input_dataframe, input_col='text', max_df=None, min_df=1, num_voca=1000, idf_weighting_scheme='inverseDocumentFrequency', norm='l2', smooth_idf=True, sublinear_tf=False, output_type=False)['model']
        
        idf_table = res['idf_table']
        tfidf_table = res['tfidf_table']
        
        print(idf_table)
        print(tfidf_table)
        
        table1 = idf_table.values.tolist()
        table2 = tfidf_table.values.tolist()
        
        self.assertListEqual(table1[0], ['air', 2.504077396776274])
        self.assertListEqual(table1[1], ['aspiring', 2.504077396776274])
        self.assertListEqual(table1[2], ['cake', 2.504077396776274])
        self.assertListEqual(table1[3], ['cool', 2.504077396776274])
        self.assertListEqual(table2[2], ['eat turkey on turkey day holiday', 'cake', 2, 0, 0.0])
        self.assertListEqual(table2[3], ['eat turkey on turkey day holiday', 'cool', 3, 0, 0.0])
        self.assertListEqual(table2[4], ['eat turkey on turkey day holiday', 'day', 4, 1, 0.4456617592757509])
        self.assertListEqual(table2[5], ['eat turkey on turkey day holiday', 'eat', 5, 1, 0.37349933584704664])
        
    def test_tfidf2(self):
        input_dataframe = self.data
        
        res = tfidf(table=input_dataframe, input_col='text', max_df=5, min_df=0, num_voca=15, idf_weighting_scheme='unary', norm='l1', smooth_idf=False, sublinear_tf=False, output_type=True)['model']
        
        idf_table = res['idf_table']
        tfidf_table = res['tfidf_table']
        
        print(idf_table)
        print(tfidf_table)
        
        table1 = idf_table.values.tolist()
        table2 = tfidf_table.values.tolist()
        
        self.assertListEqual(table1[0], ['air', 1.0])
        self.assertListEqual(table1[1], ['day', 1.0])
        self.assertListEqual(table1[2], ['eat', 1.0])
        self.assertListEqual(table1[3], ['holiday', 1.0])
        self.assertListEqual(table2[2], ['eat turkey on turkey day holiday', 'holiday', 3, 1, 0.08183872713120169])
        self.assertListEqual(table2[3], ['eat turkey on turkey day holiday', 'turkey', 14, 2, 0.16367745426240338])
        self.assertListEqual(table2[4], ['i like to eat cake on holiday', 'eat', 2, 1, 0.13429010276228892])
        self.assertListEqual(table2[5], ['i like to eat cake on holiday', 'holiday', 3, 1, 0.13429010276228892])


if __name__ == '__main__':
    unittest.main()
