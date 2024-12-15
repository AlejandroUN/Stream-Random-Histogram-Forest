import numpy as np
import random
import math
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.type_alias import AnomalyScore
import rhf_stream as rhfs

class StreamRHF(AnomalyDetector):

    def __init__(self, schema=None, number_of_trees = 100, height = 5, random_seed=1):
        super().__init__(schema, random_seed=random_seed)
        self.number_of_trees = number_of_trees
        self.height = height

    def score_instance(self, instance: Instance) -> AnomalyScore:
        return self.scores[instance.y_index]
    
    def train(self, instance: Instance):
        pass

    def predict(self, instance: Instance):
        pass

    def get_scores(self, data, N_init_pts):
        return rhfs.rhf_stream(data, self.number_of_trees, self.height, N_init_pts)