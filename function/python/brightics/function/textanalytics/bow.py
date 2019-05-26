from brightics.common.repr import BrtcReprBuilder, strip_margin, dict2MD
from brightics.function.utils import _model_dict
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters
from brightics.common.validation import raise_runtime_error
from brightics.common.classify_input_type import check_col_type

from gensim.corpora import Dictionary
import pandas as pd
import operator


def bow(table, group_by=None, **params):
    check_required_parameters(_bow, params, ['table'])
    if group_by is not None:
        return _function_by_group(_bow, table, group_by=group_by, **params)
    else:
        return _bow(table, **params)


def _bow(table, input_col, add_words=None, no_below=1, no_above=0.8, keep_n=100000, keep_tokens=None, remove_n=None):
    word_list = table[input_col].tolist()
    dictionary = Dictionary(word_list)
    if add_words != None:
        dictionary.add_documents([add_words])
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
    if remove_n != None:
        dictionary.filter_n_most_frequent(remove_n)
    
    out_table = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
    out_table.insert(loc=0, column=' ', value=dictionary.token2id.keys())
    out_table.columns = ['token', 'id']
    
    token_cnt = sorted(dictionary.dfs.items(), key=operator.itemgetter(0))
    dfs_list = []
    for i in range(len(dictionary.dfs)):
        dfs_list.append(token_cnt[i][1])
    out_table.insert(loc=2, column='document_frequency', value=dfs_list)
    
    params = { 
        'Input Column': input_col,
        'Add Words to Bag': add_words,
        'Minimum Number of Occurrence': no_below,
        'Maximum Fraction of Occurrence': no_above,
        'Keep N most Frequent': keep_n,
        'Keep Tokens': keep_tokens,
        'Remove N most Frequent' : remove_n
    }
    
    rb = BrtcReprBuilder()
    rb.addMD(strip_margin("""
        |# Bag of Words Result
        |### Parameters
        |
        | {display_params}
        |
        """.format(display_params=dict2MD(params))))
    model = _model_dict('bow')
    model['dict_table'] = out_table
    model['dictionary'] = dictionary
    model['add_words'] = add_words
    model['_repr_brtc_'] = rb.get()
    
    return {'model' : model, 'out_table': out_table}
