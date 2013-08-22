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

def compute_zmatrix(list_of_assignment):
    N = len(list_of_assignment[0])
    z = np.zeros((N, N), dtype=np.uint32)
    
    for a in list_of_assignment:
        for i in range(N):
            for j in range(N):
                if a[i] == a[j]:
                    z[i, j] +=1 
    return z

def crp_draw(N, alpha):
    group_counts = []
    assignments = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        if i == 0:
            group_counts.append(1)
            a = 0
        else:
            scores = np.zeros(len(group_counts) + 1)
            scores[:-1] = np.array(group_counts)
            scores[-1] = alpha
            p = scores/(i + alpha)
            idx = die_roll(p)
            if idx == len(group_counts):
                group_counts.append(1)
            else:
                group_counts[idx] += 1
            assignments[i] = idx
    return assignments
    

def logistic(x, mu, lamb):
    return 1.0/(1 + np.exp((x - mu)/lamb))

def sigmoid(x, mu, lamb):
    return -(x-mu) / (lamb + np.abs(x-mu))*0.5  + 0.5


def compute_purity_ratios(clustering, truth):
    """
    For a given assignment vector, compute for each true cluster types, 
    how many different clusters it was in. 
    
    Sort by true cluster size, returning: 
    
    """
    
    true_order = np.unique(truth)
    true_sizes = np.zeros_like(true_order)
    true_lut = {}
    for ti, t in enumerate(true_order):
        true_lut[t] = ti
    for t in truth:
        true_sizes[true_lut[t]] +=1

    a = np.argsort(true_sizes)
    true_order = true_order[a][::-1]
    true_sizes = true_sizes[a][::-1]

    fracs_order = []
    for ti, true_class in enumerate(true_order):
        pos = np.argwhere(truth ==true_class).flatten()
        clustered_vals = clustering[pos]

        N = len(clustered_vals)
        cv_unique = np.unique(clustered_vals)
        fracs = np.zeros(len(cv_unique))

        for ci, c in enumerate(cv_unique):
            n = len(np.argwhere(clustered_vals == c))
            fracs[ci] = float(n) / N
        fracs = sorted(fracs)[::-1]
        fracs_order.append(fracs)
    
    return true_order, true_sizes, fracs_order

def compute_purity(clustering, truth):
    """
    Returns, for each cluster in the clustering assignment vector, 
    a list of how many of each of the others there are

    """
    results = {}
    for c in np.unique(clustering):
        other_vals = truth[clustering == c]
        d = {}
        for v in other_vals:
            if v not in d:
                d[v] = 0
            d[v] +=1

        results[c] = d
    return results

def linear_dist(x, p, mu):
    """
    compute our linear distance func
    """


    y =  - x * p/mu + p 
    y[x > mu] = 0
    return y

