import numpy as np
from matplotlib import pylab
import cPickle as pickle
from copy import deepcopy
from .. import util
from .. import models
from .. import observations
NODE_POS_DTYPE = [('class',  np.uint32), 
                  ('pos' ,  np.float32, (3, ))]

def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

def example():
    """
    Create an example graph with the indicated latent classes and the location-specific
    connectivity

    # there are three graph classes, and node i connects to node J if: 
    # class
    

    """
    CLASSN = 4

    # create the cells in grids; 
    GRID_N = 8
    DIST_X = 1.0
    DIST_Y = 1.0
    DIST_Z = 1.0
    nodes = np.zeros(CLASSN * GRID_N * GRID_N, dtype=[('class',  np.uint32), 
                                                      ('pos' ,  np.float32, (3, ))])

    NODEN  = len(nodes)
    
    ni = 0
    for c in range(CLASSN):
        for xi in range(GRID_N):
            for yi in range(GRID_N):
                x = xi * DIST_X
                y = yi * DIST_Y
                z = c * DIST_Z
                nodes[ni]['class'] = c
                nodes[ni]['pos'][:] = (x, y, z)
                ni += 1

    # BAD IDEA but whatever: wire things up which is horribly N^2
    def node_pred(n1, n2):
        c1 = n1['class']
        pos1 = n1['pos']
        c2 = n2['class']
        pos2 = n2['pos']

        p = 0.001
        # if c1 == 0 and c2 == 1:
        #     if d(pos1, pos2) < 4:
        #         p = 0.4
        # elif c1 == 1 and c2 == 2:
        #     if d(pos1, pos2) > 3 and d(pos1, pos2) < 6:
        #         p = 0.2
        # elif c1 == 2 and c2 == 3:
        #     p = 0.05
        # elif c1 == 3 and c2 == 1:
        #     p = max(1.0 - d(pos1, pos2) / 5., 0.0)
        if c1 == 0 and c2 == 1:
            p = 0.4
        elif c1 == 1 and c2 == 2:
            p = 0.2
        elif c1 == 2 and c2 == 3:
            p = 0.05
        elif c1 == 3 and c2 == 0:
            p=0.7
        return np.random.rand() < p

    connectivity = np.zeros((NODEN, NODEN), dtype=np.bool)
    for ni in range(NODEN):
        for nj in range(NODEN):
            connectivity[ni, nj] = node_pred(nodes[ni], nodes[nj])
    
    return  nodes, connectivity

def generate():
    np.random.seed(0)
    nodes, connectivity = example()
    NODEN = len(nodes)
    print connectivity.shape
    pylab.subplot(1, 2, 1)
    pylab.imshow(connectivity, cmap=pylab.cm.binary, interpolation='nearest')
    c = connectivity.copy()
    c = c[np.random.permutation(NODEN)]
    c = c[:, np.random.permutation(NODEN)]
    pylab.subplot(1, 2, 2)
    pylab.imshow(c, cmap=pylab.cm.binary, interpolation='nearest')

    pylab.show()

    pickle.dump({'nodes' : nodes, 
                 'connectivity' : connectivity}, 
                open('data.pickle', 'w'))

def connect(nodes, pred, obsmodel=None):
    """
    nodes: vector of class, 3pos nodes
    pred: a function of (c1, pos1, c2, pos2) that returns prob of this connection
    
    """
    NODEN = len(nodes)
    if obsmodel == None:
        obsmodel = observations.Bernoulli()
    
    connectivity = np.zeros((NODEN, NODEN), dtype=obsmodel.dtype)
    for ni in range(NODEN):
        for nj in range(NODEN):
            params = pred(nodes[ni]['class'], 
                          nodes[ni]['pos'], 
                          nodes[nj]['class'], 
                          nodes[nj]['pos'])
            connectivity[ni, nj] = obsmodel.sample(params)
        
    return  connectivity
    
def add_class(nodes, classnum):
    NODEN = len(nodes)
    nodes_class = np.zeros(NODEN, dtype= NODE_POS_DTYPE)
    nodes_class['class'] = classnum
    nodes_class['pos'] = nodes
    return nodes_class

def prior_generate(latent, data):
    """
    Take in a partially-specified prior and latent and then 
    fill in the missing pieces. 
    
    requires: data must specify latent names, relational structure, n
    
    warning: SUFFSTATS ARE ALWAYS DRAWN FROM THE PRIOR. 
    if data and latent are specified but suffstats are not, we
    will happily fill with bullshit suffstats (that is, suffstats that 
    are drawn from the prior, and may have no relation to group structure)
    
    """ 
    
    new_latent = deepcopy(latent)
    new_data = deepcopy(data)
    
    # structural
    def cou(d, key, val): # cou
        if key not in d:
            d[key] = val

    cou(new_latent, 'domains', {})
    cou(new_latent, 'relations', {})


    for domain_name in new_data['domains']:
        cou(new_latent['domains'], domain_name, {})
        new_alpha = np.random.uniform(1.0, 5.0)
        cou(new_latent['domains'][domain_name], 
                      'hps', {} )
        cou(new_latent['domains'][domain_name]['hps'], 
            'alpha', new_alpha )
        
        alpha_val = new_latent['domains'][domain_name]['hps']['alpha']
        a = util.crp_draw(new_data['domains'][domain_name]['N'], alpha_val)
        cou(new_latent['domains'][domain_name], 
            'assignment', a)
        
    #### YOUR THINKING ABOUT SUFFICIENT STATISTICS AND PARAMETERS IS CONFUSED
    #### THE SUFFSTATS ARE UNIQUELY DETERMINED BY DATA/ASSIGNMENT IN CONJ MODELS
    #### BUT NOT IN NONCONJ MODELS 
    for rel_name, rel in new_data['relations'].iteritems():
        model_obj = models.NAMES[rel['model']]()
        cou(new_latent['relations'], rel_name, {})
        mod_new_hps = model_obj.sample_hps() 

        cou(new_latent['relations'][rel_name], 'hps', mod_new_hps)
        
        if 'ss' not in new_latent['relations'][rel_name]:
            rel_def = new_data['relations'][rel_name]['relation']
            grouplist = [np.unique(new_latent['domains'][dom]['assignment']) for dom in rel_def]
            coords = util.cart_prod(grouplist)
            ss = {}
            for c in coords:
                ss[c] = model_obj.sample_param(new_latent['relations'][rel_name]['hps'])

            new_latent['relations'][rel_name]['ss'] = ss

        if 'data' not in new_data['relations'][rel_name]:
            # generate the matrix
            data = np.zeros([new_data['domains'][dn]['N'] for dn in rel['relation']], 
                            dtype = model_obj.data_dtype())
            

            # now optionally the data
            for pos in util.cart_prod([range(new_data['domains'][dn]['N']) for dn in rel['relation']]):
                coords = [new_latent['domains'][dn]['assignment'][p] for dn, p in zip(rel['relation'], pos)]
        
                d = model_obj.sample_data(new_latent['relations'][rel_name]['ss'][tuple(coords)], 
                                          new_latent['relations'][rel_name]['hps'])
                data[pos] = d
    
            new_data['relations'][rel_name]['data'] = data

    return new_latent, new_data
    

if __name__ == "__main__":
    generate()
