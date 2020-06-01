import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
    def __init__(self, s:int, b:float, d:int):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass
        
    
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples
   
        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        x = np.asarray(features)
        xd = x[:,self.d]
        N = np.size(x,0)
        
        out = np.ones([N])*(-self.s)
        out[xd>self.b] = self.s
    
        return out.tolist()
