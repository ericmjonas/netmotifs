from nose.tools import *
import util

def test_unitue_qxes_pos():
    
    # T1 x T2 
    uap = util.unique_axes_pos([0], 3, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(uap, set([(3, 0), (3, 1), (3, 2)]))

    uap = util.unique_axes_pos([1], 1, ([0, 1, 2, 3], [0, 1, 2]))
    assert_equal(uap, set([(0, 1), (1, 1), (2, 1), (3, 1)]))

    # T1 x T1
    uap = util.unique_axes_pos([0, 1], 1, ([0, 1, 2, 3], [0, 1, 2, 3]))
    assert_equal(uap, set([(1, 0), (1, 1), (1, 2), (1, 3),
                           (0, 1), (2, 1), (3, 1)]))
    
