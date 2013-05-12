from nose.tools import *
import util
import numpy as np

from numpy.testing import assert_approx_equal

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
    

def test_crp_post_pred():
    assert_approx_equal(util.crp_post_pred(0, 1, 1.0), np.log(1.0))
    assert_approx_equal(util.crp_post_pred(0, 1, 10.0), np.log(1.0))
    assert_approx_equal(util.crp_post_pred(100, 101, 10.0), np.log(100./(100+10.0)))

def test_crp_score():
    np.random.seed(0)
    N = 100
    alpha = 1.0
    for i in range(100):
        a = np.random.randint(0, i+1, N)
        c = util.assign_to_counts(a)
        s = util.crp_score(c, alpha)
    # FIXME better tests of CRP 
