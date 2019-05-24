from brightics.common.repr import BrtcReprBuilder, strip_margin, dict2MD
from brightics.function.utils import _model_dict
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters
from brightics.common.validation import raise_runtime_error

import pandas as pd
import numpy as np
from gensim import matutils
import scipy.sparse


def term_term_mtx(table, model, group_by=None, **params):
    check_required_parameters(_term_term_mtx, params, ['table', 'model'])
    if '_grouped_data' in model:
        return _function_by_group(_term_term_mtx, table, model, group_by=group_by, **params)
    else:
        return _term_term_mtx(table, model, **params)


def _term_term_mtx(table, model, input_col, result_type='sparse'):
    corpus = table[input_col].tolist()
    
    dictionary = model['dictionary']
     
    bow_corpus = []
    for doc in corpus:
        bow_corpus.append(dictionary.doc2bow(doc))
    
    csr_matrix = matutils.corpus2csc(bow_corpus).T
    csr_matrix.data = np.array([1 for _ in range(len(csr_matrix.data))])
    
    term_term = csr_matrix.T @ csr_matrix
    term_idx_list_1 = []
    term_idx_list_2 = []
    cnt_list = []
    for i, (b, e) in enumerate(zip(term_term.indptr, term_term.indptr[1:])):
        for idx in range(b, e):
            j = term_term.indices[idx]
            if i < j:
                d = term_term.data[idx]
                term_idx_list_1.append('{}'.format(dictionary[i]))
                term_idx_list_2.append('{}'.format(dictionary[j]))
                cnt_list.append(d)
     
    if model['add_words'] is None:
        model['add_words'] = []       
    num_origin = len(dictionary) - len(model['add_words'])
    terms = [term for term in dictionary.token2id.keys()][:num_origin]
    
    if result_type == 'sparse':
        out_table = pd.DataFrame(term_idx_list_1, columns=['term1'])
        out_table['term2'] = term_idx_list_2
        out_table['number_of_documents_containing_terms'] = cnt_list
    elif result_type == 'dense':
        out_table = pd.DataFrame(term_term.todense())
        out_table.insert(loc=0, column=' ', value=terms)
        out_table.columns = np.append(" ", terms)
    else:
        raise_runtime_error("Please check 'result_type'.")
        
    rb = BrtcReprBuilder()
    model = _model_dict('term_term_mtx')
    model['term_term_mtx'] = term_term
    model['_repr_brtc_'] = rb.get()
    
    return {'out_table' : out_table}
