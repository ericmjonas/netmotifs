from nose.tools import * 


import pyirm
import util

def test_cart_prod():
    x = pyirm.cart_prod([5, 8, 12])
    set_x = set(tuple(a) for a in x)
    assert_equal(len(x), 5*8*12)
    assert_equal(len(x), len(set_x))
    assert_equal(set_x, set(util.cart_prod([range(5), range(8), range(12)])))

    
def test_unique_axes_pos():
    # T1 x T2 
    uap = pyirm.unique_axes_pos([0], 3, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(set(uap), set([(3, 0), (3, 1), (3, 2)]))

    uap = util.unique_axes_pos([1], 1, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(set(uap), set([(0, 1), (1, 1), (2, 1), (3, 1)]))

    # T1 x T1
    uap = util.unique_axes_pos([0, 1], 1, ([0, 1, 2, 3], [0, 1, 2, 3]))
    assert_equal(set(uap), set([(1, 0), (1, 1), (1, 2), (1, 3),
                           (0, 1), (2, 1), (3, 1)]))
    
