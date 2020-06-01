import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            
	    kmeans_init = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids,rk_,i_ = kmeans_init.fit(x)
            assert len(centroids.shape)==2 and centroids.shape[0]==self.n_cluster, 'means shape incorrect'
            
            Pi_k = np.zeros((self.n_cluster))
            sigma = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                nk = np.where(rk_==k)
                Nk = np.size(nk)
                
                Pi_k[k] = Nk/N

                xk = np.squeeze(x[nk,:])
                dk = xk - centroids[k,:]
                assert np.shape(dk) == (Nk,D), 'shape of xk-mu is incorrect' 
                sigma[k,:,:] = np.dot(dk.T,dk)/Nk

            self.means = centroids
            self.variances = sigma
            self.pi_k = Pi_k
                
                

        elif (self.init == 'random'):
            self.means = np.random.rand(N,D)
            identity = np.eye(D)
            variance = np.zeros((self.n_cluster, D, D))
            variance[0:self.n_cluster,:,:] = identity
            self.variances = variance
            self.pi_k = np.ones(self.n_cluster)*(1/self.n_cluster)

        else:
            raise Exception('Invalid initialization provided')

        
        l = self.compute_log_likelihood(x)
        for ite in range(self.max_iter):
            print(ite, l)            
            # E step
            gamma_ik = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                detsig = np.linalg.det(self.variances[k,:,:])
                while (detsig==0):
                    self.variances[k,:,:] = self.variances[k,:,:] + 0.001*np.eye(D)
                    detsig = np.linalg.det(self.variances[k,:,:])
                    
                
                const = 1/(np.sqrt((np.power(2*np.pi,D)) * detsig))
                
                diff = x - self.means[k,:]
                exp_arg2 = np.dot(np.linalg.inv(self.variances[k,:,:]), diff.T)
                exp_arg1 = np.multiply(diff, exp_arg2.T)
                exp_arg = np.sum(exp_arg1, axis=1)*(-0.5)
                
                density = const*np.exp(exp_arg)
                
                gamma_ik[:,k] = self.pi_k[k]*density

            determinant = np.sum(gamma_ik,axis=1)
            gamma = np.divide(gamma_ik, np.expand_dims(determinant, axis=1))

            # M step
            Nk = np.sum(gamma, axis=0)
            self.pi_k = Nk/N
            
            for k in range(self.n_cluster):
                self.means[k,:] = np.dot(gamma[:,k].T, x)/ Nk[k]

                diff = x - self.means[k,:]
                first_term = np.multiply(np.expand_dims(gamma[:,k],axis=1), diff)
                #first_term = np.dot(gamma[:,k].T, diff)
                numerator = np.dot(first_term.T, diff)
                self.variances[k,:,:] = numerator/Nk[k]

            # log likelihood
            lnew = self.compute_log_likelihood(x)
            if (np.abs(l-lnew)<self.e):
                break
            else:
                l = lnew                
        return ite
            
                
                
            
            
            
           

            
        
        

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        which_gauss = np.random.choice(self.n_cluster, N, p=self.pi_k)
        
        D = np.size(self.means,1)
        samples = np.zeros((N,D))
        for n in range(N):
            mean_ = self.means[which_gauss[n],:]
            sigma = self.variances[which_gauss[n],:,:]
            samples[n,:] = np.random.multivariate_normal(mean_,sigma)


        
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        N, D = x.shape
        
        log_likelihood = 0.0
        for n in range(N):
            log_arg = 0.
            for k in range(self.n_cluster):
                detsig = np.linalg.det(variances[k,:,:])
                while (detsig==0):
                    variances[k,:,:] = variances[k,:,:] + 0.001*np.eye(D)
                    detsig = np.linalg.det(variances[k,:,:])
                
                const = pi_k[k]/(np.sqrt(np.power((2*np.pi),D) * detsig))
                
                exp_arg1 = np.dot((x[n,:]-means[k,:]), np.linalg.inv(variances[k,:,:]))
                exp_arg2 = np.dot(exp_arg1, (x[n,:]-means[k,:]).T)
                exp_term = np.exp(-0.5*exp_arg2)
                
                log_arg = log_arg + (const*exp_term)
        
            log_likelihood = log_likelihood + np.log(log_arg)
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            raise Exception('Impliment Guassian_pdf __init__')

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            raise Exception('Impliment Guassian_pdf getLikelihood')
            return p
