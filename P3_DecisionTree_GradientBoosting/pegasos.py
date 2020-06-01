import json
import numpy as np

###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension weight vector
    - lamb: lambda used in pegasos algorithm

    Return:
    - obj_value: the value of objective function in SVM primal formulation
    """
    first_term = 0.5*lamb*(np.linalg.norm(w)**2)
    
    Xw = np.dot(X,w)
    yXw = y*np.squeeze(Xw)
    yXw1 = 1 - yXw
    yXw1[yXw1<0]=0
    second_term = sum(yXw1)/np.size(y)   
 
    
    obj_value = first_term + second_term

    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the total number of iterations to update parameters

    Returns:
    - learnt w
    - train_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    
      
    train_obj = []
    log = np.zeros((max_iterations,2))
    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
        
        X_at = Xtrain[A_t,:]
        y_at = ytrain[A_t]
                
        X_atW = np.dot(X_at,w)
        yX_atW = y_at*np.squeeze(X_atW)
        
        atplus = np.where(yX_atW<1)
        X_atpl = X_at[atplus]
        y_atpl = y_at[atplus]

        eta = 1/(lamb*iter)
        
        first_term = (1 - (eta*lamb))*w
        sum_yX = np.sum((X_atpl.T*y_atpl), axis=1, keepdims=True)
        second_term = sum_yX*(eta/k)
        w2 = first_term + second_term 

        w1 = min(1, ((1/np.sqrt(lamb)) / (np.linalg.norm(w2) )))*w2

        w = w1
        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))
                
        #print(train_obj[iter-1])
        log[iter-1,0] = iter
        log[iter-1,1] = train_obj[iter-1]
        
        
        """
        xw = np.dot(Xtrain[A_t,:], w)
        print('xw is' + str(np.shape(xw)))
        ywx = np.multiply(ytrain[A_t], np.squeeze(xw))
        print('ywx is ' + str(np.shape(ywx)))
        A_tplus = np.where(ywx<1)   # index of negative values in mini-batch
        print('A_tplus is ' +str(np.shape(A_tplus)))        

        eta_t = 1./(lamb*iter)
        
        sec_term = (np.sum(Xtrain[A_t[A_tplus],:].T*ytrain[A_t[A_tplus]], axis=1))*(eta_t/k)
        sec_term_re = np.reshape(sec_term, [np.size(sec_term),1])
        
        fir_term = (1 - (eta_t*lamb))*w
        w_tby2 = fir_term + sec_term_re
        
        w_t1 = min(1, ((1/np.sqrt(lamb)) /(np.linalg.norm(w_tby2))) )*w_tby2
        
        w = w_t1
        #print(np.linalg.norm(w))
        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))
        """
        
    #print('-----------------------------')
    
    
    
    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w_l):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
 
    Returns:
    - test_acc: testing accuracy.
    """
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    
    
    pred = np.sign(np.dot(Xtest, w_l))
    pred[pred==0] = -1

    n_correct = sum(np.equal(np.squeeze(pred), ytest).astype(int))
    n_test = np.size(ytest)
    
    test_acc = n_correct/n_test

    
    return test_acc



def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        
        
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
