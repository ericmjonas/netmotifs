import numpy as np

import synth
import tesselate
def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))

def one_class_neighbors(SIDE_N, PROX=2.0, JITTER = 0.0, CONN_PROB=0.8, 
                        cust_pred = None):
    """
    A single class; each node connects to it's neighbors closer than 
    a certain distance
    
    Planar

    returns nodes, connectivity

    """ 
    GRID_SPACE_X = 1.0
    GRID_SPACE_Y = 1.0
    GRID_N_X = SIDE_N
    GRID_N_Y = SIDE_N
    
    nodes = tesselate.grid(GRID_N_X, GRID_N_Y, 
                           GRID_SPACE_X, GRID_SPACE_Y)

    nodes += np.random.rand(len(nodes), 3) * JITTER

    nodes_with_class = synth.add_class(nodes, 0)
    

    def node_pred(c1, pos1, c2, pos2):
        if dist(pos1, pos2) < PROX:
            return CONN_PROB
        return 0.0

    if cust_pred :
        pred = cust_pred
    else:
        pred = node_pred
    connectivity = synth.connect(nodes_with_class, pred)
    
    return nodes_with_class, connectivity

def c_class_neighbors(SIDE_N, class_connectivity, 
                      JITTER = 0.0, rand_conn_prob = 0.01):

    """
    for each of c classes, we create a grid of SIDE_N x SIDE_N
    and then use the class_connectivity dictionary to set up 
    threshold connectivity

    class_connectivity = {(c1, c2) : (threshold, prob), 
                          }
    PROX: proximitiy of neighbors
    
    Planar

    returns nodes, connectivity

    """ 
    GRID_SPACE_X = 1.0
    GRID_SPACE_Y = 1.0
    GRID_N_X = SIDE_N
    GRID_N_Y = SIDE_N
    
    CLASS_N = np.max(np.array([class_connectivity.keys()]).flatten()) + 1
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

    def node_pred(c1, pos1, c2, pos2):

        for (c1_t, c2_t), (thold, prob) in class_connectivity.iteritems():
            if (c1 == c1_t) and (c2 == c2_t):
                d = dist(pos1, pos2)
                if d < thold:
                    return prob
            
        return rand_conn_prob

    connectivity = synth.connect(nodes, node_pred)

    return nodes, connectivity


def two_class_neighbors(SIDE_N, PROX=2.0, JITTER = 0.0, CONN_PROB=0.8, 
                        cust_pred = None):
    """
    Two classes: each only connects with nodes of the OTHER class less
    than the distance away. Two slightly-offset grids
    
    PROX: proximitiy of neighbors
    
    Planar

    returns nodes, connectivity

    """ 
    GRID_SPACE_X = 1.0
    GRID_SPACE_Y = 1.0
    GRID_N_X = SIDE_N
    GRID_N_Y = SIDE_N
    
    nodes_c1 = tesselate.grid(GRID_N_X, GRID_N_Y, 
                              GRID_SPACE_X, GRID_SPACE_Y)
    nodes_c2 = tesselate.grid(GRID_N_X, GRID_N_Y, 
                              GRID_SPACE_X, GRID_SPACE_Y)

    nodes_c2[:, 0] += GRID_SPACE_X/2
    nodes_c2[:, 1] += GRID_SPACE_Y/2

    nodes_c1 += np.random.rand(len(nodes_c1), 3) * JITTER
    nodes_c2 += np.random.rand(len(nodes_c2), 3) * JITTER

    n_c1 = synth.add_class(nodes_c1, 0)
    n_c2 = synth.add_class(nodes_c2, 1)

    nodes = np.hstack([n_c1, n_c2])
    print nodes.shape
    def node_pred(c1, pos1, c2, pos2):
        if (c1 == 0 and c2 == 1):
            if dist(pos1, pos2) < PROX:
                return CONN_PROB
        elif (c1 == 1 and c2 == 0): 
            if dist(pos1, pos2) < PROX:
                return CONN_PROB
            
        return 0.0

    if cust_pred :
        pred = cust_pred
    else:
        pred = node_pred
    connectivity = synth.connect(nodes, pred)
    
    return nodes, connectivity
