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

def count(x):
    """
    count the number of each item in x
    """
    v = {}
    for q in x:
        if q not in v:
            v[q] = 0
        v[q] +=1
    return v

def log_bernoulli(heads, tails, p):
    pass

def log_norm_dens(x, mu, var):
    c = -np.log(np.sqrt(var*2*np.pi) )
    v = -(x-mu)**2 / (2*var)
    return c + v 


def canonicalize_assignment(assignments):
    """
    Canonicalize an assignment vector. this works as follows:
    largest group is group 0, 
    next largest is group 1, etc. 

    For two identically-sized groups, The lower one is
    the one with the smallest row
    
    """
    groups = {}
    for gi, g in enumerate(assignments):
        if g not in groups:
            groups[g] = []
        groups[g].append(gi)
    orig_ids = np.array(groups.keys())
    sizes = np.array([len(groups[k]) for k in orig_ids])

    unique_sizes = np.sort(np.unique(sizes))[::-1]
    out_assign = np.zeros(len(assignments), dtype=np.uint32)

    # unique_sizes is in big-to-small
    outpos = 0
    for size in unique_sizes:
        # get the groups of this size
        tgt_ids = orig_ids[sizes == size]
        minvals = [np.min(groups[tid]) for tid in tgt_ids]
        min_idx = np.argsort(minvals)
        for grp_id  in tgt_ids[min_idx]:
            out_assign[groups[grp_id]] = outpos
            outpos +=1 
    return out_assign
