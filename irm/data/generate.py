import numpy as np
from .. import util 
from .. import observations

import synth
import tesselate
def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


def create_nodes_grid(SIDE_N, CLASS_N, JITTER):

    GRID_SPACE_X = 1.0
    GRID_SPACE_Y = 1.0
    GRID_N_X = SIDE_N
    GRID_N_Y = SIDE_N
    
    X_DELTA = GRID_SPACE_X/CLASS_N
    
    all_nodes = []
    for c in range(CLASS_N):
        
        nodes = tesselate.grid(GRID_N_X, GRID_N_Y, 
                               GRID_SPACE_X, GRID_SPACE_Y)
        nodes[:, 0] += X_DELTA * c

        if JITTER > 0:
            nodes += np.random.normal(0, JITTER, size=(len(nodes), 3))
        
        n_c = synth.add_class(nodes, c)
        all_nodes.append(n_c)

    nodes = np.hstack(all_nodes)
    return nodes

def c_class_neighbors(SIDE_N, class_connectivity, 
                      JITTER = 0.0, default_param=0.01, obsmodel=None):

    """
    for each of c classes, we create a grid of SIDE_N x SIDE_N
    and then use the class_connectivity dictionary to set up 
    threshold connectivity

    class_connectivity = {(c1, c2) : (threshold, param), 
                          }
    PROX: proximitiy of neighbors
    
    Planar

    returns nodes, connectivity -- note that connectivity is really observations

    """ 
    
    CLASS_N = np.max(np.array([class_connectivity.keys()]).flatten()) + 1

    nodes = create_nodes_grid(SIDE_N, CLASS_N, JITTER)

    def node_pred(c1, pos1, c2, pos2):

        for (c1_t, c2_t), (thold, params) in class_connectivity.iteritems():
            if (c1 == c1_t) and (c2 == c2_t):
                d = dist(pos1, pos2)
                if d < thold:
                    return params
            
        return default_param 

    observations = synth.connect(nodes, node_pred, obsmodel)

    return nodes, observations

def c_mixed_dist_block(SIDE_N, class_connectivity, 
                       JITTER = 0.0, rand_conn_prob = 0.01, 
                       obsmodel = None):

    """
    Mixed membership 

    class_connectivity = {(c1, c2) : ('p', prob), 
                          (c2, c3) : ('d', dist, prob)
                          }

    """ 
    
    CLASS_N = np.max(np.array([class_connectivity.keys()]).flatten()) + 1

    nodes = create_nodes_grid(SIDE_N, CLASS_N, JITTER)

    def node_pred(c1, pos1, c2, pos2):

        for (c1_t, c2_t), v in class_connectivity.iteritems():
            if (c1 == c1_t) and (c2 == c2_t):
                if v[0] == 'p' :
                    return v[1]
                else :
                    thold = v[1]
                    prob = v[2]
                    d = dist(pos1, pos2)
                    if d < thold:
                        return prob
            
        return rand_conn_prob

    connectivity = synth.connect(nodes, node_pred, obsmodel)

    return nodes, connectivity

def c_bump_dist_block(SIDE_N, class_connectivity, 
                      JITTER = 0.0, rand_conn_prob = 0.01, 
                      p_min = 0.01, obsmodel=None):

    """
    Mixed membership 

    class_connectivity = {(c1, c2) : (dist, prob, width), 
                          }

    """ 
    
    CLASS_N = np.max(np.array([class_connectivity.keys()]).flatten()) + 1

    nodes = create_nodes_grid(SIDE_N, CLASS_N, JITTER)

    def node_pred(c1, pos1, c2, pos2):

        for (c1_t, c2_t), v in class_connectivity.iteritems():
            if (c1 == c1_t) and (c2 == c2_t):
                mu = v[0]
                prob = v[1]
                width = v[2] 


                d = dist(pos1, pos2)

                return util.norm_dist_bump(d, mu, width, prob, p_min)             
        return rand_conn_prob

    connectivity = synth.connect(nodes, node_pred, obsmodel)

    return nodes, connectivity
