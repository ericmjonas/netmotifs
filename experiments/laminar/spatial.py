import numpy as np
from matplotlib import pylab


"""
config: Generate a spatially-homogeneous (x, y) area of fixed radius

[class weight vector]
[class weight vector]
[class weight vector]

"""

def sample_uniform_radius(radius):
    """
    Rejection sample
    """
    x = radius * 2
    y = radius * 2
    while np.sqrt(x**2 + y**2) > radius:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
    return x, y

    
def generate_layered_cells(N, layer_thickness, radius, 
                           layer_desc):
    """
    layer_desc : {'weight' : blah, 
    'layers' : []}
    """
    data = []
    LAYER_N = len(layer_desc['weight'])
    assert LAYER_N == len(layer_desc['layers'])

    for i in range(N):
        layer_i = die_roll(layer_desc['weight'])
        
        layer_w = layer_desc['layers'][layer_i]
        
        class_i = irm.util.die_roll(layer_w)
        
        x, y = sample_uniform_radius(radius)
        
        z = np.random.uniform(0, layer_thickness)
    
        data.append({'layer' : layer_i, 
                     'class' : class_i, 
                     'x' : x, 
                     'y' : y, 
                     'z' : z})
        
    return pandas.DataFrame(data)

def connect_laminar(cell_df, conn_prog):
    """
    Return a dataframe of from_i, to_i

    {((from_layer, from_class), 
       (to_layer, to_class), distance_thold, conn_prob)}
    
    """

    
    

