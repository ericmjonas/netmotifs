import itertools
import numpy as np

def cart_prod(list_of_objs):
    """
    Return the cartesian product of items. Does not
    check for duplicates
    """

    return itertools.product(*list_of_objs)

def unique_axes_pos(axis_pos, val, axes_possible_vals):
    """
    Return a guaranteed-unique list of coordinates with 
    val fixed at axis_pos. 

    This is the "return possible vals" query
    
    """
    def pred(x):
        for p in axis_pos:
            if x[p] == val:
                return True
        return False
    a = itertools.ifilter(pred, cart_prod(axes_possible_vals))
    return set(a)


def crp_post_pred(mk, N, alpha):
    """
    Prob of adding a customer to the table which currently seats
    mk people.
    N is total number of people INCLUDING current to-sit person
    """
    if mk == 0:
        return np.log(alpha / (N - 1 + alpha))
    else:
        return np.log(mk / (N -1 + alpha))

def assign_to_counts(assign_vect):
    """
    non-canonical assignment vector -> table sizes
    """
    d = {}
    for v in assign_vect:
        if v not in d:
            d[v] = 0
        d[v] +=1
    counts = np.zeros(len(d), dtype=np.uint32)
    for ki, k in enumerate(d.keys()):
        counts[ki] = d[k]
    return counts
        
def crp_score(counts, alpha):
    """
    Total crp score of a count vector
    """
    score = 0
    i = 0
    for table in counts:
        for customer in range(table):
            score += crp_post_pred(customer, i + 1, alpha)
            i += 1
    return score

def die_roll(v):
    """
    Take in a vector of probs and roll
    """
    x = np.cumsum(v)
    r = np.random.rand()
    return np.searchsorted(x, r)

def scores_to_prob(x):
    """
    Take in a vector of scores
    normalize, log-sumpadd, and return
    
    """
    xn = x - np.max(x)
    a = np.logaddexp.accumulate(xn)[-1]
    xn = xn - a
    return np.exp(xn)

def sample_from_scores(scores):
    return die_roll(scores_to_prob(scores))

def kl(p, q):
    idx = p > 0
    
    a = np.log(p[idx]) - np.log(q[idx])
    return np.sum(a * p[idx])
