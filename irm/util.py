import itertools

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


            
