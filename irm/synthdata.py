import numpy as np

def create_T1T2_bb(t1_n, t2_n, t1_class, t2_class):
    """
    
    """
    latent_class_matrix = np.random.rand(t1_class, t2_class)
    t1_assign = np.arange(t1_n) % t1_class
    t2_assign = np.arange(t2_n) % t2_class
    
    obs_matrix = np.zeros((t1_n, t2_n), dtype=np.bool)
    
    for ti in range(t1_n):
        for tj in range(t2_n):
            p = latent_class_matrix[t1_assign[ti], t2_assign[tj]]
            obs_matrix[ti, tj] = np.random.rand() < p

    return t1_assign, t2_assign, obs_matrix, latent_class_matrix

