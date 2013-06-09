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



"""
Workflow:
1. generate data
2. generate N inits
3. Run for 100 iterations
4. Save results
"""

