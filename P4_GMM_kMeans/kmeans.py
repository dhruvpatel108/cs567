import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        
        # Initialization
        clust_ind = np.random.choice(N, self.n_cluster)
        centroids = x[clust_ind,:]
        J = np.power(10,10)
        
               
        # Update
        for iter in range(self.max_iter):
            # distance matrix from clusters
            dist = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                dist[:,k] = np.linalg.norm(x - centroids[k,:], axis=1)
            
            # Calculate J    
            r_k = np.argmin(dist, axis=1)
            Jnew = 0
            for k in range(self.n_cluster):
                nk = np.where(r_k==k)
                dist_k = dist[nk,k]        
                Jnew = Jnew + np.square(np.linalg.norm(dist_k))
            
            Jnew = Jnew/N
            if (np.abs(J-Jnew)<self.e):
                break
            else:
                J = Jnew
            
            # Update centroids    
            for k in range(self.n_cluster):
                nk = np.where(r_k==k)
                if (np.size(nk)==0):
                    pass
                else:
                    centroids[k,:] = np.sum(np.squeeze(x[nk,:]), axis=0)/np.size(nk)
                    
            
        return (centroids, r_k, iter)        

        
        
class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        
        
        kmeans_train = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, r_k, it = kmeans_train.fit(x)
        
        centroid_labels = np.zeros((self.n_cluster))
        for k in range(self.n_cluster):
            n_k = np.where(r_k==k)
            y_k = y[n_k]
            if (np.size(n_k)==0):
                pass
            else:
                centroid_labels[k] = np.argmax(np.bincount(y_k).astype(int))
  
            
            
        
                

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        test_dist = np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            test_dist[:,k] = np.linalg.norm(x - self.centroids[k,:], axis=1)
            
        rk_test = np.argmin(test_dist, axis=1)
        assert rk_test.shape==(N,), 'argmin test shape shound be = N'
        labels = np.zeros((N))
        for k in range(self.n_cluster):
            nk = np.where(rk_test==k)
            labels[nk] = self.centroid_labels[k]
        
        return labels

