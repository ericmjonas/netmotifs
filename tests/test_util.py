from nose.tools import *
from irm import util
import numpy as np

from numpy.testing import assert_approx_equal, assert_array_almost_equal

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

def test_compute_purity_ratios():
    truth = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.uint32)
    clustering = np.array([6, 7, 8, 9, 9, 9, 1], dtype=np.uint32)
    
    true_order, true_sizes, fracs_order = util.compute_purity_ratios(clustering, truth)
    np.testing.assert_array_equal(true_order, [2, 1, 0])
    np.testing.assert_array_equal(true_sizes, [4, 2, 1])
    
    assert_array_almost_equal(fracs_order[0], [0.75, 0.25])
    assert_array_almost_equal(fracs_order[1], [0.5, 0.5])
    assert_array_almost_equal(fracs_order[2], [1.0])
    

def test_boundaries():
    a = ['a', 'b', 'c']
    b = util.get_boundaries(a)
    assert_equal(b['a'], [0, 1])
    assert_equal(b['b'], [1, 2])
    assert_equal(b['c'], [2, 3])


    a = ['a', 'a', 'b', 'c', 'c']
    b = util.get_boundaries(a)
    assert_equal(b['a'], [0, 2])
    assert_equal(b['b'], [2, 3])
    assert_equal(b['c'], [3, 5])

    a = ['a', 'a', 'b', 'c', 'c', 'd']
    b = util.get_boundaries(a)
    assert_equal(b['a'], [0, 2])
    assert_equal(b['b'], [2, 3])
    assert_equal(b['c'], [3, 5])
    assert_equal(b['d'], [5, 6])

def test_multi_napsack():
    v = [3, 5, 2, 1]
    a = util.multi_napsack(2, v)
    assert_equal([[1, 3], [0, 2]], a)
    
    a = util.multi_napsack(3, v)
    assert_equal([[1], [0], [2, 3]], a)
