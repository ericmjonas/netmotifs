from nose.tools import * 
import numpy as np


import irm
from irm import util

def test_cart_prod():
    x = irm.cart_prod([5, 8, 12])
    set_x = set(tuple(a) for a in x)
    assert_equal(len(x), 5*8*12)
    assert_equal(len(x), len(set_x))
    assert_equal(set_x, set(util.cart_prod([range(5), range(8), range(12)])))

    # test with singletons
    x = irm.cart_prod([5, 1, 1])
    set_x = set(tuple(a) for a in x)
    assert_equal(len(x), 5*1*1)
    assert_equal(len(x), len(set_x))
    assert_equal(set_x, set(util.cart_prod([range(5), range(1), range(1)])))

    
def test_unique_axes_pos():
    # T1 x T2 
    uap = irm.unique_axes_pos([0], 3, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(set(uap), set([(3, 0), (3, 1), (3, 2)]))

    uap = irm.unique_axes_pos([1], 1, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(set(uap), set([(0, 1), (1, 1), (2, 1), (3, 1)]))

    # T1 x T1
    uap = irm.unique_axes_pos([0, 1], 1, ([0, 1, 2, 3], [0, 1, 2, 3]))
    assert_equal(set(uap), set([(1, 0), (1, 1), (1, 2), (1, 3),
                           (0, 1), (2, 1), (3, 1)]))

    

    # T1 x T1
    uap = irm.unique_axes_pos([0, 1], 1, ([1], [1]))
    assert_equal(len(uap), 1)


def test_create_component():
    # can we merely create the dang thing
    x = np.zeros((10, 20, 30), dtype=np.bool)
    cc = irm.create_component_container(x.tostring(), 
                                        x.shape, "BetaBernoulli")
    assert_equal(cc.dpcount(), 10*20*30)
