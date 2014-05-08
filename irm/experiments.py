import cPickle as pickle
import numpy as np
import copy
import os, glob
import time
import cStringIO as StringIO
import irm
import irm.data
import util
import tempfile

import multyvac

"""
Code to run experiments -- this is shared across a lot of them. This is NOT
production-quality code by any means and is a grab-bag of "crap we use" 

It's only here because it's easy to import. 

"""


def create_init(latent_filename, data_filename, out_filenames, 
                init= None, keep_ground_truth=True):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)

    # FIXME : add ability to init multiple domains
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    irm_data = pickle.load(open(data_filename, 'r'))
    irm_latents = []

    rng = irm.RNG()

    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    for c, out_f in enumerate(out_filenames):
        print "generating init", out_f
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        d_N = len(latent['domains']['d1']['assignment'])
        if init['type'] == 'fixed':
            group_num = init['group_num']

            a = np.arange(d_N) % group_num
            a = np.random.permutation(a)

        elif init['type'] == 'crp':
            alpha = init['alpha']
            a = irm.util.crp_draw(d_N, alpha)
            a = np.random.permutation(a) 
        elif init['type'] == 'truth':
            a = latent['domains']['d1']['assignment']
            
        else:
            raise NotImplementedError("Unknown init type")
            
        if (not keep_ground_truth) or (c > 0) : # first one stays the same
            latent['domains']['d1']['assignment'] = a

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)
        print "estimating suffstats for %s" % out_f
        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=2)
        print "ss estimation done for ", out_f

        pickle.dump(irm.irmio.get_latent(irm_model), open(out_f, 'w'))


def plot_latent(latent, dist_matrix, 
                latent_filename, 
                ground_truth_assign = None, 
                truth_comparison_filename = None, 
                model='LogisticDistance', PLOT_MAX_DIST=200, MAX_CLASSES=20):
    """ just getting this code out of pipeline"""
    from matplotlib import pylab

    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    a = np.array(latent['domains']['d1']['assignment'])

    if ground_truth_assign == None:
        f = pylab.figure(figsize= (24, 24))
        gs = gridspec.GridSpec(1, 1)
        ax = pylab.subplot(gs[0])
    else:
        f = pylab.figure(figsize= (24, 26))
    
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,12])
        ax_types = pylab.subplot(gs[0])        
        ax = pylab.subplot(gs[1])


    print "plot_t1t1_latent"
    if "istance" in model:
        ai = irm.plot.plot_t1t1_latent(ax, dist_matrix['link'], a)
    else:
        ai = irm.plot.plot_t1t1_latent(ax, dist_matrix, a)

    if ground_truth_assign != None:
        for i in  np.argwhere(np.diff(a[ai]) != 0):
            ax_types.axvline(i, c='b', alpha=0.7, linewidth=1.0)

        ax_types.scatter(np.arange(len(ground_truth_assign)), 
                         ground_truth_assign[ai], edgecolor='none', c='k', 
                         s=2)

        ax_types.set_xlim(0, len(ground_truth_assign))
        ax_types.set_ylim(0, 80)
        ax_types.set_xticks([])
        ax_types.set_yticks([])

    f.tight_layout()
    f.suptitle(latent_filename)
    pp = PdfPages(latent_filename)
    f.savefig(pp, format='pdf')

    # SUCH A HACK SHOULD PASS IN DATA. GOD THIS CODE IS TURNING TO SHIT
    # UNDER TIME PRESSURE

    f2 =  pylab.figure(figsize= (24, 24))
    print "plot_t1t1_params"
    irm.plot.plot_t1t1_params(f2, dist_matrix, a, 
                              latent['relations']['R1']['ss'], 
                              latent['relations']['R1']['hps'], 
                              MAX_DIST=PLOT_MAX_DIST, model=model, 
                              MAX_CLASSES=MAX_CLASSES   )

    f2.tight_layout()
    f2.savefig(pp, format='pdf')

    pp.close()

    if ground_truth_assign != None:

        f = pylab.figure()
        ax_types = pylab.subplot(1, 1, 1)
        irm.plot.plot_purity_ratios(ax_types, a, ground_truth_assign)


        f.savefig(truth_comparison_filename)


def cluster_z_matrix(z, INIT_GROUPS=100, crp_alpha=5.0, beta=0.1,
                     ITERS=4, method='dpmm_bb'):

    N = len(z)
    # create the data
    if method == 'dpmm_bb':
        model = "BetaBernoulli"
        assert z.dtype == np.bool
        hps = {'alpha' : beta, 
               'beta' : beta}

    elif method == "dpmm_gp":
        model = "GammaPoisson"
        assert z.dtype == np.uint32
        hps = {'alpha': 2.0, 'beta' : 2.0}

    else:
        raise NotImplementedError("unknown method")

    data = {'domains' : {'d1' : {'N' : N}}, 
            'relations' : {'R1' : {'relation' : ('d1', 'd1'), 
                                   'model' : model, 
                                   'data' : z}}}

    latent_init = {'domains' : {'d1' : {'assignment' : np.arange(N) % INIT_GROUPS, 
                                        'hps' : {'alpha' : crp_alpha}}}, 
                   'relations' : {'R1' : {'hps' : hps}}}


    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(data, rng=rng)

    irm.irmio.set_model_latent(irm_model, latent_init, rng=rng)


    run = irm.runner.Runner(latent_init, data, 
                            irm.runner.default_kernel_config())
    run.run_iters(ITERS)

    state = run.get_state()
    return irm.util.canonicalize_assignment(state['domains']['d1']['assignment'])


def to_bucket(filename, VOLUME):
    """ 
    take the filename on disk and put it in the VOLUME
    
    """
    print "putting VOLUME=", VOLUME, "filename =", filename
    
    vol = multyvac.volume.get(VOLUME)
    #check size of file 
    if os.path.getsize(filename) < 2.5e6:
       vol.put_file(filename, filename)
    else:
        vol.sync_up(filename, filename)

def from_bucket(filename, VOLUME):
    """

    """
    vol = multyvac.volume.get(VOLUME)

    fid = tempfile.NamedTemporaryFile()
    n = fid.name
    print "syncing down"
    vol.sync_down(filename, n)
    print "reading"
    read_fid = open(n, 'r')
    data = read_fid.read()

    
    return data

def from_bucket_unpickle(filename, VOLUME):
    """
    assumes we are getting a pickled object. Whoops.
    """
    obj = pickle.loads(from_bucket(filename, VOLUME))

def inference_run(latent_filename, 
                  data_filename, 
                  kernel_config_filename,  
                  ITERS, seed, VOLUME_NAME, init_type=None, 
                  fixed_k = False, 
                  latent_samp_freq=20, 
                  relation_class = "Relation", 
                  cores = 1):
    """
    For running on the cloud
    """

    latent = from_bucket_unpickle(latent_filename, VOLUME_NAME)
    data = from_bucket_unpickle(data_filename, VOLUME_NAME)
    kernel_config = from_bucket_unpickle(kernel_config_filename, VOLUME_NAME)
    if relation_class == "Relation":
        relation_class = irm.Relation
    elif relation_class == "ParRelation":
        relation_class = irm.ParRelation
    else:
        raise NotImplementedError("unknown relation class %s" % relation_class)

    if cores == 1:
        threadpool = None
    else:
        print "Creating threadpool with", cores, "cores"
        threadpool = irm.pyirm.ThreadPool(cores)

    chain_runner = irm.runner.Runner(latent, data, kernel_config, seed, 
                                     fixed_k = fixed_k, 
                                     relation_class = relation_class,
                                     threadpool = threadpool)

    if init_type != None:
        chain_runner.init(init_type)

    scores = []
    times = []
    latents = {}
    def logger(iter, model, res_data):
        print "Iter", iter
        scores.append(model.total_score())
        times.append(time.time())

        if iter % latent_samp_freq == 0:
            latents[iter] = chain_runner.get_state(include_ss=False)
    chain_runner.run_iters(ITERS, logger)
        
    return scores, chain_runner.get_state(), times, latents

def plot_chains_hypers(f, chains, data):
    from matplotlib import pylab

    CHAINN = len(chains)
    RELATIONS = data['relations'].keys()
    per_r_hp = {}
    per_r_hp_ax = {}
    hp_n = 0
    for r in RELATIONS:
        m = data['relations'][r]['model']
        per_r_hp[r] = []
        per_r_hp_ax[r] = []
        if m == 'BetaBernoulli':
            per_r_hp[r].append('alpha')
            hp_n +=1 
            per_r_hp[r].append('beta')
            hp_n +=1 

        elif m == 'GammaPoisson':
            per_r_hp[r].append('alpha')
            hp_n +=1 
            per_r_hp[r].append('beta')
            hp_n +=1 
        elif m == 'LogisticDistance':
            for p in ['mu_hp', 'lambda_hp', 'p_min', 'p_max']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'LogisticDistanceFixedLambda':
            for p in ['mu_hp', 'lambda', 'p_min', 'p_scale_alpha_hp', 'p_scale_beta_hp']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'NormalDistanceFixedWidth':
            for p in ['mu_hp', 'p_alpha', 'p_beta', 'p_min', 'width']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'SquareDistanceBump':
            for p in ['mu_hp', 'p_alpha', 'p_beta', 'p_min']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'ExponentialDistancePoisson':
            for p in ['rate_scale_hp', 'mu_hp']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'LogisticDistancePoisson':
            for p in ['rate_scale_hp', 'mu_hp', 'lambda']:
                per_r_hp[r].append(p)
                hp_n +=1 
        elif m == 'NormalInverseChiSq':
            for p in ['mu', 'kappa', 'nu', 'sigmasq']:
                per_r_hp[r].append(p)
                hp_n +=1 

        else:
            raise RuntimeError("Unknown model'%s'" % m)
    pos = 1
    for r in RELATIONS:   
        per_r_hp_ax[r] = [] 
        for hp_name in per_r_hp[r]:
            per_r_hp_ax[r].append(pylab.subplot2grid((1+hp_n, 1), (pos, 0)))
            pos += 1
    ax_crp_alpha = pylab.subplot2grid((1+hp_n, 1), (0, 0))

    ### Plot scores
    for di, d in enumerate(chains):
        ki = sorted(d['latents'].keys())
        alpha_x_jitter = 0.1
        alpha_y_jitter = 1.0
        alphas = np.array([d['latents'][k]['domains']['d1']['hps']['alpha'] for k in ki])
        y_jitter = np.random.normal(0, alpha_y_jitter, size=len(alphas))
        ax_crp_alpha.scatter(ki, alphas + y_jitter, edgecolor='none', 
                             alpha=0.2)
        ax_crp_alpha.grid(1)
        for ri, rel_name in enumerate(per_r_hp.keys()):
            print "rel_name", rel_name
            for hp_i, hp_name in enumerate(per_r_hp[rel_name]):
                print "hp_name=", hp_name
                ax = per_r_hp_ax[rel_name][hp_i]
                vals = np.array([d['latents'][k]['relations'][rel_name]['hps'][hp_name] for k in ki])
                print vals, ki
                min_val = np.min(vals)
                max_val = np.max(vals)
                range_mid = (min_val + max_val)/2. 
                range_val = max_val - min_val
                #ax.set_ylim(range_mid - range_val, 
                #            range_mid + range_val)
                y_jitter = np.random.normal(0, 1, size=len(vals)) * (range_val * 0.05)

                ax.scatter(ki, vals + y_jitter, edgecolor='none', 
                           alpha=0.2)
                ax.set_ylabel("%s : %s" % (rel_name, hp_name), 
                             fontsize=6)
                ax.grid(1)
                ax.ticklabel_format(style='plain', axis='y', scilimits=(-8, 8))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(6) 

                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(6) 
