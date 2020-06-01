import numpy as np
import json

###### Q5.1 ######
def compute_distances(Xtrain, X):
    """
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
    """
    n_train = np.size(Xtrain,0)
    n_test = np.size(X,0)
    dists = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            dists[i,j] = np.sqrt( np.sum( (X[i,:]-Xtrain[j,:])**2 ))
            #dists[j,i] = dists[i,j] 

    return dists
	

###### Q5.2 ######
def predict_labels(k, ytrain, dists):
    """
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i]. 
    """
    n_test = np.size(dists,0)
    ypred = np.zeros((n_test))
    for i in range(n_test):
        idx = np.argpartition(dists[i,:], k)
        y_closest = ytrain[idx[0:k]]
        y_bin = np.bincount(y_closest)
        ypred[i] = np.argmax(y_bin)
    return ypred
	

###### Q5.3 ######
def compute_accuracy(y, ypred):
    """
	Compute the accuracy of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the 
	  prediction of the ith test point.
	Returns:
	- acc: The accuracy of prediction (scalar).
    """
    ypred_int = np.int32(ypred)
    n_equal = np.sum(ypred_int == y) 
    acc = n_equal/np.size(ypred,0)
    return acc
	

###### Q5.4 ######
def find_best_k(K, ytrain, dists, yval):
    """
	Find best k according to validation accuracy.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the highest validation accuracy.
	- validation_accuracy: A list of accuracies of different ks in K.
    """
    validation_accuracy = np.zeros((np.size(K)))
    for n_neighbor in range(np.size(K)):
        k = K[n_neighbor]

        n_test = np.size(dists,0)
        ypred = np.zeros((n_test))
        for i in range(n_test):
            idx = np.argpartition(dists[i,:], k)
            y_closest = ytrain[idx[0:k]]
            y_bin = np.bincount(y_closest)
            ypred[i] = np.argmax(y_bin)
            
            
        ypred_integer = np.int32(ypred)
        n_equals = np.sum(ypred_integer == yval)
        
        validation_accuracy[n_neighbor] = n_equals/np.size(ypred,0)
    
    best_k = K[np.argmax(validation_accuracy)]
    return best_k, validation_accuracy

    
	

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]
	
	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)
	
	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest
	
def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)
	
	#==================Compute distance matrix=======================
	K=[1, 3, 5, 7, 9]	
	
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	
	dists = compute_distances(Xtrain, Xval)
	
	#===============Compute validation accuracy when k=5=============
	k = 5
	ypred = predict_labels(k, ytrain, dists)
	acc = compute_accuracy(yval, ypred)
	print("The validation accuracy is", acc, "when k =", k)
	
	#==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K, ytrain, dists, yval)
    
	
	#===============test the performance with your best k=============
	dists = compute_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_accuracy = compute_accuracy(ytest, ypred)
	
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()
	
if __name__ == "__main__":
	main()
