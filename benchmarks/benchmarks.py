import numpy as np
import irm
from irm import runner, irmio, pyirmutil
import copy
from ruffus import * 
import cPickle as pickle
import pandas

"""
Inference testing of small-sized data, but across all types
and relation classes. 

"""

ITERS_TO_RUN = 2
MODELS =  [
    #"BetaBernoulliNonConj", 
    'LogisticDistance', 
    #       'LogisticDistanceFixedLambda', 
    #'LinearDistance', 
    #        'SigmoidDistance', 
    #        'GammaPoisson',
    #        'BetaBernoulli', 
    #        'NormalDistanceFixedWidth', 

    #'ExponentialDistancePoisson',
    #        'LogisticDistancePoisson', 
    # 'NormalInverseChiSq'
    #'SquareDistanceBump',  # NOTE THIS IS SO NOT READY FOR PRODUCTION the delta prob is broken
    #"MixtureModelDistribution"
]



KERNELS = {#'default_nonconj' : irm.runner.default_kernel_nonconj_config(), 
           'default_anneal_1':  irm.runner.default_kernel_anneal(1.0) # default annealing kernel that runs everything
}


RELATION_CLASSES = { 'relation' :  pyirmutil.Relation, 
                     'parrelation':  pyirmutil.ParRelation}

def benchmark_params(): # the _a is just for filtering
    np.random.seed(0)

    for GROUP_N in  [5, 10, 20, 30, 50]:
        for ENTITIES_PER_GROUP in [10, 20, 50]: # 100]:
            N = GROUP_N * ENTITIES_PER_GROUP


            latent = {'domains' : 
                      {'d1' : 
                    {'assignment' : np.random.permutation(np.arange(N) % GROUP_N)}, 
                   }}

            np.random.seed(0)
            for model_name in MODELS:
                for kernel_name, kernel_config in KERNELS.iteritems():
                    for relclass, rc in RELATION_CLASSES.iteritems():
                    
                        data = {'domains' : {'d1' : {'N' : N}}, 
                                'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                                       'model' : model_name}}}
                        infile = None

                        outfile = "benchmark.%s.%s.%d.%d.%s.pickle" % (model_name, kernel_name, GROUP_N, ENTITIES_PER_GROUP, relclass)
                        seed = np.random.randint(0, 10000)
                        yield infile, outfile, model_name, latent, data, seed, kernel_name, ITERS_TO_RUN, GROUP_N, ENTITIES_PER_GROUP, relclass


@files(benchmark_params)
def run_benchmark(infile, outfile, model_name, latent, data, seed, kernel_name, iters_to_run, GROUP_N, ENTITIES_PER_GROUP, relclass):

    kernel_config = KERNELS[kernel_name]
    np.random.seed(seed)
    new_latent, new_data = irm.data.synth.prior_generate(latent, data)
    # estimate suffstats from the data

    if relclass == 'relation':
        threadpool = None
    else:
        threadpool = irm.ThreadPool(8)

    run_truth = runner.Runner(new_latent, new_data, kernel_config,
                              seed=0, 
                              relation_class=RELATION_CLASSES[relclass], 
                              threadpool = threadpool)

    irmio.estimate_suffstats(run_truth.model, run_truth.rng)


    iter_count = 0
    res = []
    def logger(iters, model, iter_res):
        out = {'model_name' : model_name, 
               'iter' : iters, 
               'seed' : seed, 'kernel_name' : kernel_name, 
               'relclass' : relclass, 
               'group_n' : GROUP_N, 'entities_per_group' : ENTITIES_PER_GROUP}

        for t, ti in iter_res['kernel_times']:
            out['kernel_time.%s' % t] = ti
        res.append(out)
    
    run_truth.run_iters(iters_to_run, logger)

    pickle.dump(res, open(outfile, 'w'))

@merge(run_benchmark, 'benchmarks.pickle')
def merge_benchmarks(infiles, outfile):

    def get_dicts(f):
        d = pickle.load(open(f))
        return d
    lists = map(get_dicts, infiles)
    df = pandas.DataFrame(sum(lists, []))
    pickle.dump(df, open(outfile, 'w'))


        
if __name__ == "__main__":
    pipeline_run([run_benchmark, merge_benchmarks])#  no multiprocess, , multiprocess=2)
