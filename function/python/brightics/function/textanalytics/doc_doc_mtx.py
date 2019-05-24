from brightics.common.repr import BrtcReprBuilder, strip_margin, dict2MD
from brightics.function.utils import _model_dict
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters
from brightics.common.validation import raise_runtime_error

import pandas as pd
import numpy as np
from gensim import corpora, matutils
import scipy.sparse


def doc_doc_mtx(table, model, group_by=None, **params):
    check_required_parameters(_doc_doc_mtx, params, ['table', 'model'])
    if '_grouped_data' in model:
        return _function_by_group(_doc_doc_mtx, table, model, group_by=group_by, **params)
    else:
        return _doc_doc_mtx(table, model, **params)


def _doc_doc_mtx(table, model, input_col, result_type='sparse'):
    corpus = table[input_col].tolist()

    dictionary = model['dictionary'] 
    
    bow_corpus = []
    for doc in corpus:
        bow_corpus.append(dictionary.doc2bow(doc))
        
    csr_matrix = matutils.corpus2csc(bow_corpus).T
    csr_matrix.data = np.array([1 for _ in range(len(csr_matrix.data))])
    doc_doc = csr_matrix @ (csr_matrix.T)
    doc_idx_list_1=[]
    doc_idx_list_2=[]
    cnt_list = []
    for i, (b, e) in enumerate(zip(doc_doc.indptr, doc_doc.indptr[1:])):
        for idx in range(b, e):
            j = doc_doc.indices[idx]
            if i < j:
                d = doc_doc.data[idx]
                doc_idx_list_1.append('doc_{}'.format(i))
                doc_idx_list_2.append('doc_{}'.format(j))
                cnt_list.append(d)
            
    doc_idx = ['doc_{}'.format(i) for i in range(len(corpus))]
    
    if result_type == 'sparse':
        out_table = pd.DataFrame(doc_idx_list_1, columns=['document1'])
        out_table['document2']=doc_idx_list_2
        out_table['number_of_common_terms']=cnt_list
    elif result_type == 'dense':
        out_table = pd.DataFrame(doc_doc.todense())
        out_table.insert(loc=0, column=' ', value=doc_idx)
        out_table.columns = np.append("", doc_idx)
    else:
        raise_runtime_error("Please check 'result_type'.")
    
    rb = BrtcReprBuilder()
    model = _model_dict('doc_doc_mtx')
    model['input_col'] = input_col
    model['doc_doc_mtx'] = doc_doc
    model['_repr_brtc_'] = rb.get()
    
    return {'out_table' : out_table}
