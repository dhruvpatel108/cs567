import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return
        
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        
        string = ''
        for idx_cls in range(node.num_cls):
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the index of the feature to be split

        self.feature_uniq_split = None # the possible unique values of the feature to be split


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array, 
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of 
                      corresponding training samples 
                      e.g.
                                  ○ ○ ○ ○
                                  ● ● ● ●
                                ┏━━━━┻━━━━┓
                               ○ ○       ○ ○
                               ● ● ● ●
                               
                      branches = [[2,2], [4,0]]
            '''
            branch_arr = np.array(branches)
            n_class = np.size(branch_arr,0)
            n_branch = np.size(branch_arr,1)
            cross_h = np.zeros((1,n_branch))

            cross_entropy = 0.0
            for j in range(n_branch):
                prob_branch = branch_arr[:,j]/np.sum(branch_arr[:,j])
                assert len(prob_branch)==n_class, 'dim mismatch'
                for i in range(n_class):
                    if (prob_branch[i]==0):
                        cross_h[0,j] = cross_h[0,j] + 0     # to avoid log(0) error..
                    else:
                        cross_h[0,j] = cross_h[0,j] - prob_branch[i]*np.log2(prob_branch[i])

                cross_entropy = cross_entropy + ((np.sum(branch_arr[:,j])/np.sum(branch_arr))*cross_h[0,j])

            return cross_entropy





        ind_feat = 0
        labels_arr = np.asarray(self.labels)
        compare_cond = np.zeros(len(self.features[0]))
  
        for idx_dim in range(len(self.features[0])):
            feat_vector = np.asarray([self.features[n][idx_dim] for n in range(len(self.features))])
            branch_names = np.unique(feat_vector)
            branches_arr = np.zeros((self.num_cls,np.size(branch_names)))
   
            for j in range(np.size(branch_names)):
                eg_ind = np.asarray([ind for ind in range(len(feat_vector)) if feat_vector[ind]==branch_names[j]])
                label_branch = labels_arr[eg_ind]
                unq_values, counts = np.unique(label_branch, return_counts=True)
                branches_arr[unq_values, j] = counts

            compare_cond[idx_dim] = conditional_entropy(branches_arr.tolist())
   
            if (idx_dim>0):
                if (compare_cond[idx_dim]<compare_cond[ind_feat]):
                    ind_feat = idx_dim

        
        self.dim_split = ind_feat
        f_vector = np.asarray([self.features[n][self.dim_split] for n in range(len(self.features))])
        self.feature_uniq_split = np.unique(f_vector).tolist()
        
        
        
         
            
  
         #self.children = self.features 



        
        



        n_data = len(self.features)
        
        feature_master = np.asarray(self.features)
        dummy_vect = np.ones(np.size(feature_master,0))*(500)
        feature_master[:, self.dim_split] = dummy_vect



        for i_child in self.feature_uniq_split:
            
            indices = np.argwhere(f_vector==i_child)
            ind1 = np.reshape(indices, [len(indices)])
            child_labels = np.asarray(self.labels)[ind1].tolist()
            feature_child = feature_master[ind1,:].tolist()

            child_node = TreeNode(feature_child, child_labels, self.num_cls)
                
            if ((np.min(feature_child)==500) or (len(np.unique(child_labels))<2)):
                child_node.splittable=False
            else:
                child_node.splittable=True
                
            self.children.append(child_node)



        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



