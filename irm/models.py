import numpy as np
from scipy.special import betaln

class BetaBernoulli(object):
    def create_hps(self):
        """
        Return a hypers obj
        """
        return {'alpha' : 1.0, 'beta': 1.0}

    def create_ss(self):
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
            return np.log(heads + alpha) - denom
        else:
            return np.log(tails + beta) - denom

class AccumModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self):
        return {'sum' : 0}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = ss['sum'] + datum
        ss_new = self.create_ss()
        ss_new['sum'] = s
        return ss_new
    
    def ss_rem(self, ss, hp, datum):
        s = ss['sum'] - datum
        ss_new = self.create_ss()
        ss_new['sum'] = s
        return ss_new

    def ss_score(self, ss, hp):
        return ss['sum'] + hp['offset']

    def post_pred(self, ss, hp, datum):
        return datum

class VarModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self):
        """
        We could make this incremental if we wanted to
        """
        return {'dp' : []}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = list(ss['dp'])
        s.append(datum)
        ss_new = self.create_ss()
        ss_new['dp'] = s 
        return ss_new
    

    def ss_rem(self, ss, hp, datum):
        s = list(ss['dp'])
        s.remove(datum)
        ss_new = self.create_ss()
        ss_new['dp'] = s
        return ss_new

    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) + np.mean(ss['dp']) + hp['offset']

        return s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        
        return np.var(ss) - np.var(ss['dp']) + np.mean(ss) - np.mean(ss['dp'])

class NegVarModel(VarModel):
    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) +  hp['offset']

        return -s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        
        return -(np.var(ss) - np.var(ss['dp']))
