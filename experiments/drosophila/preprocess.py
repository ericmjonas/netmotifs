import pandas 
import cPickle as pickle
from ruffus import *
import numpy as np
import re


SECTION_THICKNESS = 0.042 # microns, mean, from Supplementary materials 
PIXEL_WIDTH = 37.0 / 4096 # microns, in X-Y, arrived at by taking imaging area size
                          # divided by pixel size

@files("../../../data/drosophila/nature12450-s3.xls", "data.all.pickle")
def load_excel_file(excel_file, pickle_file):

    
    n = pandas.io.excel.ExcelFile(excel_file)
    df = n.parse(u'Supplementary Table 1 - Complet', header=None, skiprows=range(3), na_values="?")
    df = df.drop(range(25, 33)) # get rid of comments
    df.columns = ['postcite', 'pre.id', 'pre.x', 'pre.y', 'pre.z', 'pre.comment', 
                  'post.id', 'post.x', 'post.y', 'post.z', 'post.comment', 'comments']

    df['pre.x'] = df['pre.x'] * PIXEL_WIDTH
    df['pre.y'] = df['pre.y'] * PIXEL_WIDTH
    df['pre.z'] = df['pre.z'] * SECTION_THICKNESS

    df['post.x'] = df['post.x'] * PIXEL_WIDTH
    df['post.y'] = df['post.y'] * PIXEL_WIDTH
    df['post.z'] = df['post.z'] * SECTION_THICKNESS

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

@files(extract_useful_cells, "celldata.pickle")
def per_cell_data(indata, outfile):
    synapses = pickle.load(open(indata, 'r'))['synapses']
    

    post_mean = synapses.groupby('post.id').mean()
    pre_mean =  synapses.groupby('pre.id').mean()

    df = pandas.concat([pre_mean[['pre.x', 'pre.y', 'pre.z']], post_mean[['post.x', 'post.y', 'post.z']]], axis=1)

    celltype_re = re.compile("(.+) (\d+)")
    df['type'] = [celltype_re.match(n).groups()[0] for n in df.index.values]

    pickle.dump({'celldata' : df}, 
                open(outfile, 'w'))


if __name__ == "__main__":
    pipeline_run([extract_useful_cells, per_cell_data])
