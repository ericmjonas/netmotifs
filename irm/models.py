import numpy as np

class BetaBernoulli(object):
    def create_hps(self):
        """
        Return a hypers obj
        """

    def create_ss(self):
        """
        """

    def data_dtype(self):
        """
        """
    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
    def ss_rem(self, ss, hp, datum):
        """
        """
    def ss_score(self, ss, hp):
        pass

    def post_pred(self, ss, hp, datum):
        pass

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
