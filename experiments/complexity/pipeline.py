from ruffus import *

import numpy as np
import cPickle as pickle
import irm
import time
import pandas
from matplotlib import pylab

def bb_gen(N):
    return np.random.rand(N, N) > 0.5


data_types = {'bb' : {'model' : "BetaBernoulli", 
                      'datafunc' : bb_gen}, 
              # 'bbnc' : {'model' : "BetaBernoulliNonConj", 
              #           'datafunc' : bb_gen},
}

ITERS = 3

def datasets():
    for dt in data_types: 
        for groups in [2, 4, 8, 12, 16, 20]:
            for dp_in_group in [2, 4, 8, 16, 24, 28, 32, 48, 64]:
                for iters in [ITERS]:
                    outfilename = "%s.%d.%d.%d" % (dt, groups, dp_in_group, iters)
                    yield None, (outfilename + ".latent", 
                                 outfilename + ".data", outfilename + ".config", 
                                 outfilename + ".meta"), dt, groups, dp_in_group, iters
                                 

@files(datasets)
def create_data(infilename, (latent_file, data_file, config_file, meta_file), 
                dt, groups, dp_in_group, inters):

    rows = groups * dp_in_group
    connectivity = data_types[dt]['datafunc'](rows)
    
    latent, data = irm.irmio.default_graph_init(connectivity, 
                                                model=data_types[dt]['model'])
    latent['domains']['d1']['assignment'] = np.arange(rows) % groups
    
    kernel_config = [('nonconj_gibbs', {'M' : 10, 'impotent': True})]

    pickle.dump(latent, open(latent_file, 'w'))
    pickle.dump(data, open(data_file, 'w'))
    pickle.dump(kernel_config, open(config_file, 'w'))
    pickle.dump({'dt' : dt, 
                 'groups' : groups, 'dp_in_group': dp_in_group}, 
                open(meta_file, 'w'))

@transform(create_data, regex(r"(.+)\.latent"), r"\1.results.pickle")
def run_data((latent_f, data_f, config_f, meta_f), outfile):
    latent = pickle.load(open(latent_f))
    data = pickle.load(open(data_f))
    config = pickle.load(open(config_f))
    meta = pickle.load(open(meta_f, 'r'))
    
    r = irm.runner.Runner(latent, data, config)
    
    times = []
    toffset = time.time()
    def logger(iter, state):
        t = time.time() - toffset
        times.append(t)
    r.run_iters(ITERS, logger) # fixme thread through
    meta.update({'times' : times})
    pickle.dump(meta, 
                open(outfile, 'w'))

@merge(run_data, "output.pickle")
def merge_runs(infiles, outfile):
    df_dicts = []
    for f in infiles:
        d = pickle.load(open(f))
        df_dicts.append(d)
    df = pandas.DataFrame(df_dicts)

    pickle.dump(df, open(outfile, 'w'))

@files(merge_runs, "plot.pdf")
def plot_times(infile, outfile):
    df = pickle.load(open(infile, 'r'))
    
    df['t.deltas'] = df['times'].apply(lambda x: np.diff(x))
    df['t.means'] = df['t.deltas'].apply(lambda x: np.mean(x))

    for dt_name, dt_grp in df.groupby('dt'):
        f = pylab.figure(figsize=(8,6))
        for grp_x, groups_grp in dt_grp.groupby('groups'):
            dp_in_group = np.array(groups_grp['dp_in_group'])**2
            means = np.array(groups_grp['t.means'])
            ai = np.argsort(dp_in_group)
            pylab.plot(dp_in_group[ai], means[ai], label=grp_x)
            pylab.text(dp_in_group[ai][-1], means[ai][-1], grp_x)
            pylab.scatter(dp_in_group, means)
        pylab.xlabel("observations per class")
        f.savefig(dt_name+ ".pdf")
    
pipeline_run([create_data, run_data, merge_runs, plot_times])
