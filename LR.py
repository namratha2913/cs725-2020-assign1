import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    def __init__(self):
    	print("Class scalar object created!")
    	return
    	raise NotImplementedError
    def __call__(self,features, is_train=False):
    	
        print("Normalizing features!")
        minm= features.min()
        maxm= features.max()
        avrg= features.mean()
        sd= features.std()

        # --Normalization ---
        '''for i in range(0,features.shape[0]):
            if (maxm-minm)!=0:
                features[i] = (features[i]-minm)/(maxm-minm)'''
            
        #---Standardisation  ---
        for i in range(0,features.shape[0]):
            if sd!=0:
                features[i] = (features[i]-avrg)/sd
        
        return features
        
        raise NotImplementedError


def get_features(csv_path,is_train=False,scaler=None):
    
    f = pd.read_csv(csv_path)
    if is_train==False :
        feature_matrix = np.array(f)
    else:   
        feature_matrix = np.array(f[f.columns[:-1]])

    if scaler!=None:
        feature_matrix = scaler(feature_matrix) 
    else:
        feature_matrix=feature_matrix/10000
    #print(feature_matrix)
    bias = np.array([1]*(f.shape[0])) #bias added
    feature_matrix = np.hstack((np.atleast_2d(bias).T,feature_matrix))
    print("feature_matrix create and returned!")

    return feature_matrix
    raise NotImplementedError

def get_targets(csv_path):
    
    f = pd.read_csv(csv_path)
    targets = np.array(f[f.columns[-1]])
    print("Target returned!")

    return targets
    raise NotImplementedError
     

def analytical_solution(feature_matrix, targets, C=0.0):

    # w * = (X^T * X+ C*I)^-1 * X^T* y
    X = feature_matrix
    X_transpose = np.transpose(X)
    mat = np.matmul(X_transpose,X)
    mat = mat+ C*np.identity(feature_matrix.shape[1])

    w_soln = np.matmul(np.matmul(np.linalg.inv(mat),X_transpose),targets)
    print(w_soln)
    print("calculated weights!")
    return w_soln
    raise NotImplementedError

def get_predictions(feature_matrix, weights):
    
    y_pred = np.matmul(feature_matrix,weights)
    #print("fetching predctions..")
    y_pred = np.rint(y_pred)
    return y_pred

    raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    
    n= len(targets)
    y_pred = get_predictions(feature_matrix,weights)
    loss = (y_pred - targets) ** 2
    mse_loss = np.sum(loss)/(n)
    print("returning the MSE_loss!")

    return mse_loss

    raise NotImplementedError

def l2_regularizer(weights):

    l2_reg = np.sum(weights**2)
    print(" sending l2_regularization value")
    return l2_reg

    raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):


    loss = mse_loss(feature_matrix,weights,targets)+C*l2_regularizer(weights)
    print("Calculating and returning loss value")
    return loss

    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    

    y_pred = get_predictions(feature_matrix,weights)
    m = len(targets)
    n = len(weights)
    gradient = [0] * n
    #print(n)
    for i in range(0,n):
        grad = 0
        #if i!=0 :
        for j in range(0,m):
            grad += ((int(y_pred[j])-targets[j])*feature_matrix[j,i]*2)

        gradient[i] = (grad/m)+(C*(2*float(weights[i])))
        #print(gradient[i])
        #else:
        #   gradient[i] = (np.sum(((y_pred-targets)*2))/m

    gradient =np.array(gradient)

    return gradient 

    raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):

    sample = np.column_stack((feature_matrix,targets))
    #print(sample)
    np.random.shuffle(sample)
    mini_batches =[]
    #print(sample))
    for i in range(0,round(len(sample)/batch_size)) :
        #print(i,)
        mini = []
        if (i+1)*batch_size>len(sample):
            mini.append(sample[i * batch_size:,:-1])
            mini.append(sample[i * batch_size:,-1])
        else:
            mini.append(sample[i * batch_size:(i + 1)*batch_size,:-1])
            mini.append(sample[i * batch_size:(i + 1)*batch_size,-1])
        
        mini_batches.append(mini)

    return mini_batches
    raise NotImplementedError
    
def initialize_weights(n):
    
    weights = np.random.randn(n)
    #weights = np.random.uniform(-10,10,n)
    #weights = np.random.normal(0,0.1,n)
    np.random.shuffle(weights)
    print("return randomly initiated bias and weights...")
    return weights

    raise NotImplementedError

def update_weights(weights, gradients, lr):
    
    weights = weights-(lr*gradients)     
    return weights

    raise NotImplementedError

def early_stopping(flag):
    # allowed to modify argument list as per your need
    # return True or False

    if flag ==2 :
        return 1
    else :
        return 0

    raise NotImplementedError
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    
    #a sample code is as follows -- 
    n_wt = train_feature_matrix.shape[1]
    weights = initialize_weights(n_wt)
    #print(weights)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    
    early_flag =0
    epoch = 0
    index = list(range(0,len(train_targets)))

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))

    for step in range(0,max_steps):

        #batch_no =0

        #sample a batch of features and gradients
        mini_batches = sample_random_batch(train_feature_matrix,train_targets,batch_size)

        for mini_batch in mini_batches: 

            #batch_no +=1 

            #print("evaluating batch no {} \t".format(batch_no))

            #print(mini_batch[0].shape)
            #compute gradients
            gradients = compute_gradients(mini_batch[0], weights, mini_batch[1], C)

            #update weights
            weights = update_weights(weights, gradients, lr)
            #print(weights)

        epoch = epoch +1
        dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
        train_loss = mse_loss(train_feature_matrix, weights, train_targets)
        print("epoch {} \t  dev loss: {} \t train loss: {}".format(epoch,dev_loss,train_loss))

    return weights



def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss



if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',True), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=0.00008)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    '''test_features = get_features('data/test.csv',False)
    predictions = get_predictions(test_features, a_solution)
    result_file = open('test_results_closed_form.csv', 'w')
    result_file.write("{},{}{}".format("instance_id","shares", '\n'))
    for i in range(len(predictions)):
        result_file.write("{},{}{}".format(i,predictions[i], '\n'))'''

    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',True,scaler), get_targets('data/dev.csv')

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features,
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.00999,
                        C=0.000000000000000955,
                        batch_size = 32,
                        max_steps=200)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    '''test_features = get_features('data/test.csv',False,scaler)
    predictions = get_predictions(test_features, a_solution)
    result_file = open('test_results_gradient descent.csv', 'w')
    result_file.write("{},{}{}".format("instance_id","shares", '\n'))
    for i in range(len(predictions)):
        result_file.write("{},{}{}".format(i,predictions[i], '\n'))'''
    


