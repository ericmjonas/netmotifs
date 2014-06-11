import numpy as np
import irm
from matplotlib import pylab
from ruffus import * 
import pandas
import cPickle as pickle
import sklearn.metrics
import sys
sys.path.append("../../../paper/code")
import models
import connattribio
import os
import glob
import multyvac
import colorbrewer

"""
Ok this needs to be more realistic

Laminar types
spatial tesselation
connetivity matrix

derive soma layers, etc. 

"""
XYSCALE = 100
ZSCALE = 50
Z_BODY_STD = 2.0
COMP_K = 3
CONTACT_Z_VAR = (Z_BODY_STD/2.)**2


INIT_CONFIGS = {'fixed_20_100' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'fixed_4_100' : {'N' : 4, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'debug_2_100' : {'N' : 2, 
                                 'config' : {'type' : 'fixed', 
                                             'group_num' : 100}}, 
            }


debug_kernel = irm.runner.default_kernel_anneal(1.0, 20)
debug_kernel[0][1]['subkernels'][-1][1]['grids']['MixtureModelDistribution'] = None

slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300
slow_anneal[0][1]['subkernels'][-1][1]['grids']['MixtureModelDistribution'] = None


long_anneal = irm.runner.default_kernel_anneal(64, 900)
long_anneal[0][1]['subkernels'][-1][1]['grids']['MixtureModelDistribution'] = None

xlong_anneal = irm.runner.default_kernel_anneal(128, 3900)
xlong_anneal[0][1]['subkernels'][-1][1]['grids']['MixtureModelDistribution'] = None

KERNEL_CONFIGS = {
    'debug_20' : {'ITERS' : 20, 
                  'kernels': debug_kernel
            }, 
    'anneal_slow_400' : {'ITERS' : 400, 
                         'kernels' : slow_anneal},
    'anneal_long_1000' : {'ITERS' : 1000, 
                         'kernels' : long_anneal},
    'anneal_xlong_10000' : {'ITERS' : 10000, 
                         'kernels' : xlong_anneal},

    }


EXPERIMENTS = [
    ('test', 'debug_2_100', 'debug_20'), 
    #('test', 'fixed_20_100', 'anneal_slow_400'), 
    #('test', 'fixed_4_100', 'anneal_long_1000'), 
    ('test', 'fixed_4_100', 'anneal_xlong_10000'), 

    #('retina.xsoma' , 'fixed_20_100', 'anneal_slow_400'), 
    #('retina.xsoma' , 'fixed_20_100', 'anneal_slow_400'), 
]

VOLUME_NAME = "connattrib_mixingtest"
MULTYVAC_LAYER = "test11"
DEFAULT_CORES = 8
DEFAULT_RELATION = "ParRelation"
WORKING_DIR = "data"

@files(None, "volume.%s.sentinel" % VOLUME_NAME)
def create_volume(infile, outfile):
    multyvac.volume.create(VOLUME_NAME, '/%s' % VOLUME_NAME)
    vol = multyvac.volume.get(VOLUME_NAME) 
    fname = "%s/dir.setinel" % WORKING_DIR
    fid = file(fname, 'w')
    fid.write("test")
    fid.close()
    vol.put_file(fname, fname)
   
    open(outfile, 'w').write("done\n")


def to_f32(x):
    a = np.array(x).astype(np.float32)
    assert type(a) == np.ndarray
    return a

def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)


def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))


def create_synapse_profile_latent(modes, max_modes):
    """
    return p, mu
    """
    mu = np.random.rand(max_modes)
    p = np.random.dirichlet(np.ones(max_modes))
    p[modes:] = 0.0
    p = p / np.sum(p)
    
    return p, mu
    
    
def sample_pos_from_profile(p, mu, N, var):
    pos = np.zeros(N)
    for i in range(N):
        k = irm.util.die_roll(p)
        assert k < len(p)
        pos[i] = np.random.normal(mu[k], np.sqrt(var))
    return pos

def create_k_synapse_profiles(K, max_modes, rate=1.0):
    params = np.zeros((K, 2, max_modes))

    
    for i in range(K):
        modes = np.max([np.random.poisson(rate) + 1, max_modes])
        params[i][0], params[i][1] = create_synapse_profile_latent(modes, max_modes)
    
    return params


def create_data():
    conn_config = generate_block_config(class_n, nonzero_frac)
    obsmodel = irm.observations.Bernoulli()
    
    nodes_with_class, connectivity = generate.c_class_neighbors(side_n, 
                                                                conn_config,
                                                                JITTER=jitter, 
                                                                obsmodel=obsmodel)
    # do we need to generate in space? I think we should generate with
    # laminarity and in space. IN space NO ONE CAN HEAR YOU MIX

def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

def tesselate_space(N):
    """
    Returns the xy positions of ~K units roughly tesselating the
    space. Note this may fail and is also O(N^2)
    """

    MAX_ITERS = N*10
    pos = np.zeros((N, 2))
    TGT_MIN_DIST = 1./ (np.sqrt(N) * 2)
    for i in range(N):
        iter = 0
        min_dist = 1e6
        while min_dist > TGT_MIN_DIST:
            candidate_position = np.random.rand(2)
            if i == 0:
                min_dist = 0

            for j in range(i):
                d = dist(pos[j], candidate_position)
                if d < min_dist:
                    min_dist =d 
            iter += 1

            if iter > MAX_ITERS:
                raise Exception("Failed to sample")

        pos[i] = candidate_position
    return pos

@files(None, "source.data")
def create_data(infile, outfile):

    np.random.seed(1)

    # horrible rejection-sampling-based 
    K = 20
    # this is upper-triangular
    params = np.zeros((K, K, 2))

    p_min = 0.01
    p_max = 0.95
    SPACE_SCALE = 10.0
    for source_k in range(K):
        for dest_k in range(source_k, K):
            # FIXME this is where we could add in bimodality

            if source_k == dest_k: # SUPPRESS SELF_LINKS
                mu = 0.01 # np.random.exponential(SPACE_SCALE)
                lamb = 0.01
            else:
                mu = np.random.exponential(SPACE_SCALE)
                lamb = mu # np.random.exponential(SPACE_SCALE)
            params[source_k, dest_k] = mu, lamb

    # this is the rough dimensions of the cube

    z_pos = np.random.rand(K) * ZSCALE


    # create the cell positions
    CELLS_PER_TYPE = 50
    CELL_N = K * CELLS_PER_TYPE
    cell_pos = np.zeros((CELL_N, 3))
    cell_types = np.zeros(CELL_N, dtype=np.uint)
    conn_mat = np.zeros((CELL_N, CELL_N), dtype=np.uint8)


    # For the time being, the synapse profile has nothing to do 
    # with the actual connectivity
    syn_profiles = create_k_synapse_profiles(K, COMP_K, 3.0)
    syn_profiles[:, 1] *= ZSCALE

    for k in range(K):
        print "Generating params for k=", k

        these_cell_pos = np.zeros((CELLS_PER_TYPE, 3))
        these_cell_pos[:, 2] = np.random.normal(z_pos[k], Z_BODY_STD, 
                                                size=CELLS_PER_TYPE)


        # now tesselate
        these_cell_pos[:, :2] = tesselate_space(CELLS_PER_TYPE)
        these_cell_pos[:, :2] *= XYSCALE

        cell_pos[k*CELLS_PER_TYPE:(k+1)*CELLS_PER_TYPE]= these_cell_pos
        cell_types[k*CELLS_PER_TYPE:(k+1)*CELLS_PER_TYPE]= k

    # permute it 
    ai = np.random.permutation(len(cell_pos))
    cell_pos = cell_pos[ai]
    cell_types = cell_types[ai]

    # now we wire up this abomination
    for i in xrange(CELL_N):
        for j in xrange(i, CELL_N):
            i_type = cell_types[i]
            j_type = cell_types[j]
            d = dist(cell_pos[i], cell_pos[j])

            if i_type < j_type:
                mu, lamb = params[i_type, j_type]
            else:
                mu, lamb = params[i_type, j_type]
            p = irm.util.logistic(d, mu, lamb)
            p = p * (p_max - p_min)  + p_min
            conn = 0
            if i == j:
                conn = 0
            elif np.random.rand() < p:
                conn = 1

            conn_mat[i, j] = conn
            conn_mat[j, i] = conn

    SYN_CONTACT_RATE = 100
    syn_contacts = []
    for i in range(CELL_N):
        k = cell_types[i]
        sp = sample_pos_from_profile(syn_profiles[k, 0], 
                                     syn_profiles[k, 1], 
                                     np.random.poisson(SYN_CONTACT_RATE) +1, 
                                     CONTACT_Z_VAR)
        print sp
        syn_contacts.append(sp.tolist())

    # create the data frame
    df = pandas.DataFrame({"type_id" : cell_types, 
                           "contact_z_list" : syn_contacts, 
                           'x' : cell_pos[:, 0], 
                           'y' : cell_pos[:, 1], 
                           'z' : cell_pos[:, 2],
                           'cell_id' : np.arange(len(cell_types))})



    pickle.dump({'cells' : df, 
                 'conn_mat' : conn_mat, 
                 'syn_profiles' : syn_profiles, 
                 'params' : params, 
                 'z_pos' : z_pos}, 
                open(outfile, 'w'))

                 
def plot():




    # cell positions

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    for k in range(K):
        these_cell_pos = cell_pos[k*CELLS_PER_TYPE:(k+1)*CELLS_PER_TYPE]
        ax.hist(these_cell_pos[:, 2])
        pylab.show()



    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)

    ax.imshow(conn_mat, interpolation='nearest', cmap=pylab.cm.Greys)


    pylab.show()
@files(create_data, [td('test.data'), td('test.latent'), td('test.meta')])
def create_data_latent(infile, (data_filename, latent_filename, 
                                meta_filename)):

    d = pickle.load(open(infile, 'r'))
    cells =  d['cells']
    conn_mat = d['conn_mat']
    dist_mats = {}

    for dim in ['x', 'y', 'z']:
        x = np.array(cells[dim])
        x.shape = (len(x), 1)
        dist_mats[dim] = sklearn.metrics.pairwise.pairwise_distances(x)

    
    graph_latent, graph_data = models.create_conn_dist_lowlevel(conn_mat, dist_mats, 
                                                                'xyz', 
                                                                'LogisticDistance')

    mulamb = 10.0

    p_max = 0.95

    HPS = {'mu_hp' : mulamb,
           'lambda_hp' : mulamb,
           'p_min' : 0.01, 
           'p_max' : p_max}

    graph_latent['relations']['R1']['hps'] = HPS



    contact_z_list = models.create_mixmodeldata(cells['contact_z_list'], 
                                                0, ZSCALE)
    feature_desc = {
        'contact_z_list' : {'data' : contact_z_list,
                            'model' : 'MixtureModelDistribution'}, 
        'soma_z' : {'data' : to_f32(cells['z']), 
                                'model' : 'NormalInverseChiSq'}, 
    }


    feature_latent, feature_data = connattribio.create_mm(feature_desc)
    feature_latent['relations']['r_contact_z_list']['hps'] = {'comp_k': COMP_K, 
                                                              'var_scale' : CONTACT_Z_VAR, 
                                                              'dir_alpha' : 1.0}

    soma_x_HPS = {'kappa' : 0.0001, 
                  'mu' : 50.0, 
                  'sigmasq' : 0.1, 
                  'nu' : 10.0}

    feature_latent['relations']['r_soma_z']['hps'] = soma_x_HPS

    latent, data = models.merge_graph_features(graph_latent, graph_data, 
                                feature_latent, feature_data, 
                                'd1')


    pickle.dump(latent, open(latent_filename, 'w'))
    pickle.dump(data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 }, open(meta_filename, 'w'))

@transform(create_data_latent, suffix(".data"), [".debug.png"])
def data_debug_plot((data_filename, latent_filename, 
                     meta_filename), (debug_plot_filename, )):

    data = pickle.load(open(data_filename, 'r'))
    
    soma_z = data['relations']['r_soma_z']['data'] 
    contact_z_list = data['relations']['r_contact_z_list']['data']
    
    # simple scatter
    allpts = []
    for cl in range(len(contact_z_list)):
        for j in range(contact_z_list[cl]['len']):
            allpts.append((cl, contact_z_list[cl]['points'][j]))
    allpts = np.array(allpts)

    

    f = pylab.figure()
    ax = f.add_subplot(1, 2, 1)

    ax.plot(soma_z)

    ax = f.add_subplot(1, 2, 2)

    ax.scatter(allpts[:, 0], allpts[:, 1], edgecolor='none', s=1, 
               alpha=0.1)

    
    f.savefig(debug_plot_filename)

def init_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)
            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


@follows(create_data_latent)            
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"
    
    irm.experiments.create_init(latent_filename, data_filename, 
                                out_filenames, 
                                init= init_config['config'], 
                                keep_ground_truth=False)

def experiment_generator():
    for data_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)

            inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
            exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
            yield [data_filename, inits], exp_name, kernel_config_name

@follows(create_volume)
@follows(create_inits)
@files(experiment_generator)
def run_exp((data_filename, inits), wait_file, kernel_config_name):
    # put the filenames in the data
    irm.experiments.to_bucket(data_filename, VOLUME_NAME)
    test = irm.experiments.from_bucket(data_filename, VOLUME_NAME)

    [irm.experiments.to_bucket(init_f, VOLUME_NAME) for init_f in inits]
    kernel_config_filename = kernel_config_name + ".pickle"

    kc = KERNEL_CONFIGS[kernel_config_name]
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    fixed_k = kc.get('fixed_k', False)
    cores = kc.get('cores', DEFAULT_CORES)
    relation_class = kc.get('relation_class', DEFAULT_RELATION)

    pickle.dump(kernel_config, open(kernel_config_filename, 'w'))

    irm.experiments.to_bucket(kernel_config_filename, VOLUME_NAME)


    CHAINS_TO_RUN = len(inits)

    
    jids = []

    for init_i, init in enumerate(inits):
        jid = multyvac.submit(irm.experiments.inference_run, 
                              init, 
                              data_filename, 
                              kernel_config_filename, 
                              ITERS, 
                              init_i, 
                              VOLUME_NAME, 
                              None, 
                              fixed_k, 
                              relation_class = relation_class, 
                              cores = cores, 
                              _name="%s-%s-%s" % (data_filename, init, 
                                                  kernel_config_name), 
                              _layer = MULTYVAC_LAYER,
                              _multicore = cores, 
                              _core = 'f2')
        jids.append(jid)


    pickle.dump({'jids' : jids, 
                'data_filename' : data_filename, 
                'inits' : inits, 
                'kernel_config_name' : kernel_config_name}, 
                open(wait_file, 'w'))



@transform(run_exp, suffix('.wait'), '.samples')
def get_results(exp_wait, exp_results):
    d = pickle.load(open(exp_wait, 'r'))
    
    chains = []
    # reorg on a per-seed basis
    for jid in d['jids']:
        job = multyvac.get(jid)
        print "getting", jid
        chain_data = job.get_result()
        
        chains.append({'scores' : chain_data[0], 
                       'state' : chain_data[1], 
                       'times' : chain_data[2], 
                       'latents' : chain_data[3]})
        
    pickle.dump({'chains' : chains, 
                 'exp' : d}, 
                open(exp_results, 'w'))

    
CIRCOS_DIST_THRESHOLDS = [10]

@transform(get_results, suffix(".samples"), 
           [(".circos.%02d.png" % d, 
             ".circos.%02d.small.svg" % d)  for d in range(len(CIRCOS_DIST_THRESHOLDS))])
def plot_circos_latent(exp_results, 
                       out_filenames):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']
    print "meta_infile=", meta_infile

    d = pickle.load(open(meta_infile, 'r'))
    conn = d['conn_mat']
    cells = d['cells']

    cell_types = cells['type_id']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    # soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
    # pos_vec = soma_positions['pos_vec'][cell_id_permutation]
    # print "Pos_vec=", pos_vec
    if 'R1' in data['relations']:
        model_name = data['relations']['R1']['model']
    else:
        model_name = None

    # this is potentially fun: get the ranges for each type
    TYPE_N = np.max(cell_types) + 1



    TGT_CMAP = pylab.cm.gist_heat
    coarse_colors = {'other' : [210, 210, 210]}
    for n_i, n in enumerate(['gc', 'nac', 'mwac', 'bc']):
        coarse_colors[n] = colorbrewer.Set1[4][n_i]
    print "THE COARSE COLORS ARE", coarse_colors
    
    for fi, (circos_filename_main, circos_filename_small) in enumerate(out_filenames):
        CLASS_N = len(np.unique(cell_assignment))
        

        class_ids = sorted(np.unique(cell_assignment))

        custom_color_map = {}
        for c_i, c_v in enumerate(class_ids):
            c = np.array(pylab.cm.Set1(float(c_i) / CLASS_N)[:3])*255
            custom_color_map['ccolor%d' % c_v]  = c.astype(int)

        colors = np.linspace(0, 360, CLASS_N)
        color_str = ['ccolor%d' % int(d) for d in class_ids]

        for n, v in coarse_colors.iteritems():
            custom_color_map['true_coarse_%s' % n] = v

        circos_p = irm.plots.circos.CircosPlot(cell_assignment, 
                                               ideogram_radius="0.5r",
                                               ideogram_thickness="30p", 
                                               karyotype_colors = color_str, 
                                               custom_color_map = custom_color_map)

        if model_name == "LogisticDistance":
            v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
                                               sample_latent['relations']['R1']['ss'], 
                                               sample_latent['relations']['R1']['hps'], 
                                               model_name)
            thold = 0.60 
            ribbons = []
            links = []
            for (src, dest), p in v.iteritems():
                if p > thold:
                    ribbons.append((src, dest, int(30*p)))
            circos_p.set_class_ribbons(ribbons)

        pos_min = 0
        pos_max = ZSCALE
        pos_r_min = 1.00
        pos_r_max = pos_r_min + 0.3
        ten_um_frac = 10.0/(pos_max - pos_min)

        circos_p.add_plot('scatter', {'r0' : '%fr' % pos_r_min, 
                                      'r1' : '%fr' % pos_r_max, 
                                      'min' : pos_min, 
                                      'max' : pos_max, 
                                      'glyph' : 'circle', 
                                      'glyph_size' : 8, 
                                      'color' : 'black',
                                      'stroke_thickness' : 0
                                      }, 
                          cells['z'], 
                          {'backgrounds' : [('background', {'color': 'vvlgrey', 
                                                            'y0' : pos_min, 
                                                            'y1' : pos_max})],  
                           'axes': [('axis', {'color' : 'vgrey', 
                                              'thickness' : 1, 
                                              'spacing' : '%fr' % ten_um_frac})]})

        # circos_p.add_plot('heatmap', {'r0' : '1.34r', 
        #                                 'r1' : '1.37r', 
        #                                 'min' : 0, 
        #                                 'max' : 72, 
        #                               'stroke_thickness' : 0, 
        #                               'color' : ",".join(true_color_list) }, 
        #                   cell_types)
        
        types_sparse = np.array(cells['type_id'], dtype=np.float32)
        types_sparse[types_sparse <72] = np.nan
 
        circos_p.add_plot('scatter', {'r0' : '1.8r', 
                                      'r1' : '1.9r', 
                                      'min' : 70, 
                                      'max' : 78, 
                                      'gliph' : 'circle', 
                                      'color' : 'black', 
                                      'stroke_thickness' : 0}, 
                          types_sparse, 
                          {'backgrounds' : [('background', {'color': 'vvlblue', 
                                                            'y0' : 70, 
                                                            'y1' : 78})],  
                           'axes': [('axis', {'color' : 'vgrey', 
                                              'thickness' : 1, 
                                              'spacing' : '%fr' % (0.1)})]})

        
        # this is aken directly from the thing 
        X_HIST_BINS = np.linspace(0, ZSCALE, 40)
        hists = np.zeros((len(cells), len(X_HIST_BINS)-1))
        for cell_i, (cell_id, cell) in enumerate(cells.iterrows()):
            h, e = np.histogram(cell['contact_z_list'], X_HIST_BINS)
            hists[cell_i] = h

        for bi, b in enumerate(X_HIST_BINS[:-1]):
            width = 0.4/20.
            start = 1.3 + width*bi
            end = start + width
            circos_p.add_plot('heatmap', {'r0' : '%fr' % start, 
                                          'r1' : '%fr' % end, 
                                          'stroke_thickness' : 0, 
                                          'color' : 'greys-6-seq'}, 
                              hists[:, bi])



        # circos_p.add_plot('scatter', {'r0' : '1.01r', 
        #                               'r1' : '1.10r', 
        #                               'min' : 0, 
        #                               'max' : 3, 
        #                               'gliph' : 'square', 
        #                               'color' : 'black', 
        #                               'stroke_thickness' : 0}, 
        #                   [type_lut[i] for i in cell_types])



        irm.plots.circos.write(circos_p, circos_filename_main)
        
        circos_p = irm.plots.circos.CircosPlot(cell_assignment, ideogram_radius="0.5r", 
                                               ideogram_thickness="80p", 
                                               karyotype_colors = color_str, 
                                               custom_color_map = custom_color_map)
        
        if model_name == "LogisticDistance":
            v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
                                               sample_latent['relations']['R1']['ss'], 
                                               sample_latent['relations']['R1']['hps'], 
                                               model_name)
            thold = 0.50 
            ribbons = []
            links = []
            for (src, dest), p in v.iteritems():
                if p > thold:
                    ribbons.append((src, dest, int(30*p)))
            circos_p.set_class_ribbons(ribbons)
                                            
        irm.plots.circos.write(circos_p, circos_filename_small)

@transform(get_results, suffix(".samples"), 
           ".chainstats.pickle")
def compute_chainstats_per(exp_results, 
                           out_filename):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']

    d = pickle.load(open(meta_infile, 'r'))
    cells = d['cells']
    cell_types = cells['type_id']
    latent_measurements = []
    iter_measurements = []

    for chain_pos, c in enumerate(chains):
        if type(c['scores']) == int:
            continue  # failed run
        
        sample_latent = c['state']
        scores = c['scores']
        times = c['times']
        cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])
        for i in range(len(c['scores'])):
            iter_measurements.append({'chain_i' : chain_pos, 
                                      'iter' : i, 
                                      'score' : scores[i], 
                                      'time' : times[i] - times[0]})
        for l_iter, l_val in c['latents'].iteritems():
            cell_assignment = np.array(l_val['domains']['d1']['assignment'])
            
            ca = irm.util.canonicalize_assignment(cell_assignment)
            


            ari = sklearn.metrics.adjusted_rand_score(cell_types, ca)
            latent_measurements.append({'chain_i' : chain_pos, 
                                        'iter' : l_iter, 
                                        'score' : scores[l_iter-1], 
                                        'ari' : ari})
    latent_df = pandas.DataFrame(latent_measurements)
    iter_df = pandas.DataFrame(iter_measurements)
    pickle.dump({'latent_df' : latent_df, 
                 'iter_df' : iter_df}, 
                open(out_filename, 'w'))

        

if __name__ == "__main__":
    pipeline_run([create_volume, 
                  create_data, 
                  create_data_latent, 
                  create_inits, 
                  get_results, 
                  plot_circos_latent, 
                  data_debug_plot, 
                  compute_chainstats_per
              ])
            
