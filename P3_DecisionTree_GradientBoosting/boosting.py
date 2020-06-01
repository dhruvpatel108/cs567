import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs      # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T
    
        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples
   
        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        C = np.array(self.clfs_picked)
        B = np.array(self.betas)
        
        run_sum = np.zeros([len(features)])

        for t in range(self.T):
            #print(t)
            #print(str(self.betas[t]))
            #print(str(self.clfs_picked[t].predict(features)))
            #print(str(np.multiply(self.clfs_picked[t].predict(features),self.betas[t])))
            run_sum = run_sum + np.multiply(self.clfs_picked[t].predict(features),self.betas[t])
            #print(run_sum)
            
            
        
        out = np.sign(run_sum)
        out[out==0] = -1

        return out.tolist()
        
        

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples
   
        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''
        x = np.array(features)
        y = np.array(labels)
        N = np.size(x,0)
        D0 = np.ones([N])*(1/N)
        D1 = np.zeros([N])
        
        
        for t in range(self.T):
            if (t==0):
                Dt = D0

            counter = 0

            h_loss = np.zeros([self.num_clf])
            
            min_loss = 0
            min_ind = 0
            for clf in self.clfs:
                             
                h = clf.predict(features)

                compare = np.equal(h,y).astype(int)                    
                compare[compare==0]=5
                compare[compare==1]=0
                compare[compare==5]=1 #compare[i] is 1 where h!=y and is 0 where h=y
                

                h_loss[counter] = np.sum(compare*Dt)
                if (counter==0):
                    min_loss = h_loss[counter]
                    h_t = clf
                else:
                    if (h_loss[counter]<min_loss):
                        min_loss = h_loss[counter]
                        h_t = clf
                        min_ind = counter
                        #print(h_t.predict(features))
                
                
                #temp[counter,:] = clf.predict(features)
                counter = counter + 1
                
                
                
            #ht_ind = np.argmin(h_loss)
            #h_t = temp[ht_ind,:]

            e_t = h_loss[min_ind]
            b_t = 0.5*np.log((1-e_t)/e_t)
            
            compare1 = np.equal(h_t.predict(features),y)
            true_ind = np.argwhere(compare1==True)
            false_ind = np.argwhere(compare1==False)
            
            D1[true_ind] = Dt[true_ind]*np.exp(-b_t)
            D1[false_ind] = Dt[false_ind]*np.exp(b_t)
            
            D1_norm = D1/np.sum(D1)
            Dt = D1_norm
            
            self.clfs_picked.append(h_t)
            self.betas.append(b_t)
            
            
        return self.clfs_picked, self.betas
            
            
                
           
  
  

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)



    
