"""
Code that combines all model-specific operations into one
object
"""

class BetaBernoulli(object):
    def __init__(self, data):
        self.data

    def data_len(self):
        return self.data

    def create_component(self, group_coords):
        """
        Create a new component at that group coord
        """
    def delete_component(self, group_coords):
        pass

    def total_score(self):
        pass

    def post_pred(self, coordinates, dp):
        pass
    
    def add_dp(self, group_coords, dp):
        pass

    def rem_dp(self, group_coords, dp):
        pass


