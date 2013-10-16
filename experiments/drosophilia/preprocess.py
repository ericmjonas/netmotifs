import pandas 
import cPickle as pickle
from ruffus import *
import numpy as np



@files("../../../data/drosophila/nature12450-s3.xls", "data.all.pickle")
def load_excel_file(excel_file, pickle_file):

    
    n = pandas.io.excel.ExcelFile(excel_file)
    df = n.parse(u'Supplementary Table 1 - Complet', header=None, skiprows=range(3), na_values="?")
    df = df.drop(range(25, 33)) # get rid of comments
    df.columns = ['postcite', 'pre.id', 'pre.x', 'pre.y', 'pre.z', 'pre.comment', 
                  'post.id', 'post.x', 'post.y', 'post.z', 'post.comment', 'comments']

    pickle.dump(df, open(pickle_file, 'w'))

@files(load_excel_file, "synapses.pickle")
def extract_useful_cells(infile, outfile):
    df = pickle.load(open(infile))
    
    # "what is a cell?" Well due to the poor description in supplementary
    # materials we're calling everything with a string (unicode) name a cell

    id_vc = df['pre.id'].value_counts()
    idx = np.array(np.array(map(type, id_vc.index.values)) == unicode)
    cells_of_interest = id_vc[idx].index.values
    print len(cells_of_interest)

    s_post = set(df['post.id'])
    s = s_post.intersection(cells_of_interest)
    print "There are", len(s), "overlapping cells with unicode names" 

    sub_df = df[df['pre.id'].isin(s)]
    sub_df = sub_df[sub_df['post.id'].isin(s)]


    pickle.dump({'synapses' : sub_df, 
                 'cell_ids' : list(s)}, 
                open(outfile, 'w'))


if __name__ == "__main__":
    pipeline_run([extract_useful_cells])
