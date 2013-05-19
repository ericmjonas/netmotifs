import numpy as np
from scipy.special import betaln

class BetaBernoulli(object):
    def create_hps(self):
        """
        Return a hypers obj
        """
        return {'alpha' : 1.0, 'beta': 1.0}

    def create_ss(self, hps):
        """
        """
        return {'heads' : 0, 'tails' : 0}

    def data_dtype(self):
        """
        """
        return np.bool

    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] +=1
        else:
            new_ss['tails'] +=1 

        return new_ss


    def ss_rem(self, ss, hp, datum):
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] -=1
        else:
            new_ss['tails'] -=1 

        return new_ss

    def ss_score(self, ss, hp):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        logbeta_a_b = betaln(alpha, beta)

        return betaln(alpha+heads,beta+tails) - logbeta_a_b; 

    def post_pred(self, ss, hp, datum):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        den = np.log(alpha+beta + heads + tails)

        if datum:
            return np.log(heads + alpha) - den
        else:
            return np.log(tails + beta) - den

class AccumModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self, hps):
        return {'sum' : 0, 
                'count' : 0}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = ss['sum'] + datum
        ss_new = self.create_ss(hp)
        ss_new['sum'] = s
        ss_new['count'] = ss['count'] + 1
        return ss_new
    
    def ss_rem(self, ss, hp, datum):
        s = ss['sum'] - datum
        ss_new = self.create_ss(hp)
        ss_new['sum'] = s
        ss_new['count'] = ss['count'] - 1
        assert ss_new['count'] >= 0

        return ss_new

    def ss_score(self, ss, hp):
        return ss['sum'] + hp['offset']

    def post_pred(self, ss, hp, datum):
        return datum

class VarModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self, hps):
        """
        We could make this incremental if we wanted to
        """
        return {'dp' : []}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = list(ss['dp'])
        s.append(datum)
        ss_new = self.create_ss(hp)
        ss_new['dp'] = s 
        return ss_new
    

    def ss_rem(self, ss, hp, datum):
        s = list(ss['dp'])
        s.remove(datum)
        ss_new = self.create_ss(hp)
        ss_new['dp'] = s
        return ss_new

    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) + np.mean(ss['dp']) + hp['offset']

        return s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        print "post_pred", s, ss, hp, datum
        return np.var(s) - np.var(ss['dp']) + np.mean(s) - np.mean(ss['dp'])

class NegVarModel(VarModel):
    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) +  hp['offset']

        return -s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        
        return -(np.var(ss) - np.var(ss['dp']))


class NonConjBetaBernoulli(object):
    """
    A non-conjugate instantaition of the beta-bernoulli conjugate
    model where we explicitly represent the P

    FIXME: Am I only keeping the sufficient statistics for MPD? 
    """
    def create_hps(self):
        """
        Return a hypers obj
        """
        return {'alpha' : 1.0, 'beta': 1.0}

    def create_ss(self, hps):
        """
        """
        p = self.sample_from_prior(hps)

        return {'p' : p, 
                'heads' : 0, 'tails' : 0}

    def data_dtype(self):
        """
        """
        return np.bool

    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] +=1
        else:
            new_ss['tails'] +=1 

        return new_ss


    def ss_rem(self, ss, hp, datum):
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] -=1
        else:
            new_ss['tails'] -=1 

        return new_ss

    def ss_score(self, ss, hp):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        logbeta_a_b = betaln(alpha, beta)

        return betaln(alpha+heads,beta+tails) - logbeta_a_b; 

    def post_pred(self, ss, hp, datum):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        den = np.log(alpha+beta + heads + tails)

        if datum:
            return np.log(heads + alpha) - den
        else:
            return np.log(tails + beta) - den
