import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io as sio
from scipy.linalg import inv
from scipy.sparse import csgraph as cg
from scipy.stats import norm
import sklearn.metrics.pairwise as kernels
from sklearn.preprocessing import StandardScaler as ssc
from sklearn.preprocessing import MinMaxScaler as msc
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.pipeline import Pipeline, make_pipeline

from IPython import embed
from collections import Counter
import csv
import time
import copy

from SVC import *

# Set up QP Problem
def SVDD_minimize(input_ss, gamma, C):
    K = kernels.rbf_kernel(input_ss, input_ss, gamma)
    f = -np.diag(K)
    H = 2 * K
    b = C
    I = np.arange(input_ss.shape[0])  
   
    [Alpha, fval] = qpssvm(H, f, b, I)
    alpha = Alpha
#     inx = (Alpha > pow(10, -5)).ravel()  
    alpha[alpha<1e-5] = 0
#     alpha = Alpha[inx]         
    return K, alpha

# to calculate the dual solution using kernels
def support(x, input_ss, K, alpha, gamma, model = None):
    
    if len(x.shape)==1:
        xx = x.reshape(1,-1)
    else:
        xx = x
    
    if model.kernel is 'rbf':        
        r = np.diag(kernels.rbf_kernel(xx, xx, gamma))-2*kernels.rbf_kernel(xx, input_ss, gamma).dot(alpha)+alpha.reshape(1,-1).dot(K).dot(alpha)
    elif model.kernel is 'polynomial':
        r = np.diag(kernels.polynomial_kernel(xx, xx, gamma))-2*kernels.polynomial_kernel(xx, input_ss, gamma).dot(alpha)+alpha.reshape(1,-1).dot(K).dot(alpha)
    
    return r

# to calculate the high dimensional feature mapping
def support_rkhs(x, input_ss, K, alpha, gamma, model):
    if len(x.shape)==1:
        xx = x.reshape(1,-1)
    else:
        xx = x
     
    if model.kernel is 'rbf':
        r = np.sum((rbf_trans(model,xx).flatten()-model.a_star)**2,axis=-1)
    elif model.kernel is 'polynomial':
        r = np.sum((support_rkhs(model,xx)-model.a_star)**2,axis=-1)

    return r

#solve SVDD by iterative gradients
def fit_SVDD_grad(X, y, gamma, C, lr, n_iter,model=None, n_components = 200, X_sep=None, y_sep=None, L=1, kappa=1):
    #(Phase I) Solve SVDD
    K, alpha = SVDD_minimize(X, gamma, C)
    model.N = X.shape[0]

    alpha = alpha.flatten()
    model.n_components = n_components
    if model.fun is support_rkhs:
        if model.kernel is 'rbf':        
            model.rbf = RBFSampler(n_components = n_components, gamma=gamma)
            model.rbf.fit(X)
            X_features = rbf_trans(model, X)
            model.a_star = np.dot(X_features.T,alpha)
            
        elif model.kernel is 'polynomial':
            raise NotImplementedError

    #(Phase I) SVDD noise
    if model.epsilon is not None and model.fun is support_rkhs:
        F = model.a_star.shape[0]
        laplace_b = 2*model.C*L*kappa*(F**0.5)/(model.epsilon) # SVDD noise
        model.a_star = model.a_star + np.random.laplace(0,laplace_b,(model.a_star.shape))
#         print("BEFORE CLIPPING",np.linalg.norm(model.a_star))        
        model.a_star = model.a_star/np.linalg.norm(model.a_star) if np.linalg.norm(model.a_star)>1 else model.a_star #clipping
#         print("CLIPPING",np.linalg.norm(model.a_star))
    
    #(Phase II) find (S)EPs
    if model.sep_grad==False:
        raise NotImplementedError
    else:
        if model.fun is support:
            raise NotImplementedError
        elif model.fun is support_rkhs:
            if X_sep is not None:
                X = X_sep
                y = y_sep
            N_locals, hyp_newlocal, hyp_newlocal_val, hyp_match_local, local, match_local = findSEP_grad_rkhs(X, X, K, alpha, gamma, model,lr,n_iter)       
    
    #(Phase II) Counter noise
    hyp_label = []
    for i in np.unique(hyp_match_local):   
        Counter_noise = []
        matches = Counter(y[hyp_match_local[match_local]==i])
        if model.epsilon is not None:
            for cc in range(np.min(y),len(np.unique(y))+np.min(y)):
                Counter_noise.append(matches[cc]+np.random.laplace(0,1/model.epsilon,1))
            hyp_label.append(np.argmax(np.array(Counter_noise)))
        else:
            hyp_label.append(Counter(y[hyp_match_local[match_local]==i]).most_common(1)[-1][0])
    hyp_label = np.array(hyp_label)
    hyp_label += np.min(y) - np.min(hyp_label) 
    sep_label = hyp_label[hyp_match_local]

    return local, sep_label, K, alpha

#calculate phi mapping in RBF kernel
def rbf_trans(model, X):
    projection = safe_sparse_dot(X, model.rbf.random_weights_)
    projection2 = copy.deepcopy(projection)
    np.cos(projection, projection)
    np.sin(projection2,projection2)
    projection *= 1.0 / np.sqrt(model.n_components)
    projection2 *= 1.0 / np.sqrt(model.n_components)
    return np.concatenate([projection,projection2],axis=-1)

#find (S)EP using gradients (sin, cos)
def findSEP_grad_rkhs(X, input_ss, K, alpha, gamma,  model=None, lr=1,  n_iter = 30):
    [N, dim] = X.shape
    N_locals = []
    local_val = []
    lr0 = lr
    model.N = N

    for i in range(N):
        x0 = copy.deepcopy(X[i].reshape(1,-1))
        lr=lr0
        for j in range(n_iter):
            if float(j/n_iter) == 0.8:
                lr = lr/5            

            tmp = np.dot(x0, model.rbf.random_weights_) 
            tmp2 = copy.deepcopy(tmp)
            np.sin(tmp, tmp)
            np.cos(tmp2, tmp2)
            new_tmp = tmp * model.rbf.random_weights_
            new_tmp2 = tmp2 * model.rbf.random_weights_ 
            grad_phi = (- new_tmp * ((1/X.shape[-1])**1/2))
            grad_phi2 =  (new_tmp2 * ((1/X.shape[-1])**1/2))
            grad_phi_concat = np.concatenate([grad_phi,grad_phi2],axis=-1)

            grad =  np.dot(grad_phi_concat,(rbf_trans(model,x0)-model.a_star).flatten())*2
            x0 -= lr * grad 

        temp = x0.reshape(-1,)
        val = float(model.fun(x0, input_ss, K, alpha, gamma,model))

        N_locals.append(temp)
        local_val.append(val)

    #integrate numerical error in finding (S)EPs
    N_locals = np.array(N_locals)
    local_val = np.array(local_val)    
    local, I, match_local = np.unique(np.round(N_locals*20), axis=0, return_index=True, return_inverse=True)
    newlocal = N_locals[I, :]
    newlocal_val = local_val[I]
       
    
    #construct hypercubes
    hyp_local, hyp_I, hyp_match_local = np.unique(np.round(newlocal/model.round_sep), axis=0, return_index=True, return_inverse=True)
    I2=[]
    for jj in range (len(np.unique(hyp_match_local))):
        index_jj = np.where(hyp_match_local==jj)[0]
        I2_jj = np.argmin(newlocal_val[index_jj])
        I2.append(index_jj[I2_jj])
    I2=np.array(I2)
    hyp_newlocal = newlocal[I2, :]
    hyp_newlocal_val = newlocal_val[I2]
    

    return [N_locals, hyp_newlocal, hyp_newlocal_val, hyp_match_local, newlocal, match_local]


#inference using gradients,(S)EPs
def inference_grad(X_test, input_ss, sep, sep_label, K, alpha, gamma, model=None, lr=1,  n_iter = 30):
    [N, dim] = X_test.shape        
    pred = []
    lr0 = lr
    model.sep_inf = []
    for i in range(N):  
        x0 = copy.deepcopy(X_test[i].reshape(1,-1))
        lr=lr0
        for j in range(n_iter):
            if float(j/n_iter) == 0.8:
                lr = lr/10            

            tmp = np.dot(x0, model.rbf.random_weights_)
            tmp2 = copy.deepcopy(tmp)
            np.sin(tmp, tmp)
            np.cos(tmp2, tmp2)
            new_tmp = tmp * model.rbf.random_weights_
            new_tmp2 = tmp2 * model.rbf.random_weights_ 
            grad_phi = (- new_tmp * ((1/X_test.shape[-1])**1/2))
            grad_phi2 =  (new_tmp2 * ((1/X_test.shape[-1])**1/2))
            grad_phi_concat = np.concatenate([grad_phi,grad_phi2],axis=-1)

            grad =  np.dot(grad_phi_concat,(rbf_trans(model,x0)-model.a_star).flatten())*2
            x0 -= lr * grad #/ N
        temp = x0.reshape(-1,)
        val = float(model.fun(x0, input_ss, K, alpha, gamma,model))       
        temp_dist = pairwise_distances([temp], sep)
        model.sep_inf.append(temp)
        pred.append(sep_label[np.argmin(temp_dist[0])])
    return pred


def inference_nearest(X, input_ss_l,  sep, sep_label, K, alpha, gamma, model=None):
    print("Inference with the nearest EP")
    [N, dim] = X.shape
    pred = []
    
    for i in range(N):
        x0 = copy.deepcopy(X[i].reshape(1,-1))
        temp_dist = pairwise_distances(x0, sep)
        s = np.sort(temp_dist)
        counts = np.bincount(np.array(sep_label)[np.argsort(temp_dist).flatten()[:model.nearest]])
        pred.append(np.argmax(counts))
    return pred


class SVDD(object):
    
    def __init__(self, kernel='linear', C=0, gamma=1, degree=3, sep_grad = False, 
                 fun=support, lr = 1e-3, n_iter = 10, epsilon = 0, L=1, round_sep=0.1, nearest= None):

        if C is None:
            C=0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        C = float(C)
        gamma = float(gamma)
        degree=int(degree)

        self.C = C
        self.gamma = gamma
        self.degree = degree
        # *** kernel ***
        self.kernel = kernel
        
        self.lr = lr
        self.n_iter = n_iter
        # *** fun=support -> dual kernel trick, support_rkhs -> phi calcuate ***
        # support_cond, support_rkhs_cond
        self.fun = fun
        
        #epsilon for dp, L for LLF
        self.epsilon = epsilon
        self.L = L
        self.round_sep = round_sep
        
        self.sep_grad=sep_grad
        self.nearest = nearest #

        
    def fit(self, X, y, n_components = 200, X_sep=None,y_sep=None):
        self.X_train = X
        self.n_components = n_components        
        self.local, self.sep_label, self.K, self.alpha = fit_SVDD_grad(X, y, self.gamma, self.C, self.lr, self.n_iter, self,n_components, X_sep,y_sep, L=self.L)
           
    def predict(self, X):
        if self.nearest is not None:
            pred = inference_nearest(X, self.X_train, self.local, self.sep_label, self.K, self.alpha, self.gamma, self)
        elif ((self.fun is support_rkhs) and self.sep_grad is True):
            pred = inference_grad(X, self.X_train, self.local, self.sep_label, self.K, self.alpha, self.gamma, self, self.lr, self.n_iter*2)
        return pred
    

def main():
    acc_list=[]
    accs = []
    mean_stds = []

    acc_list.append(['data','acc','epsilon','C','gamma','lr','#seps','training time','test time'])

    data_names = ['three_moons','five_Gaussians','iris','wine','vowel','satimage','segment','shuttle_class3']
    Cs = [0.05, 0.1, 0.05, 0.05, 0.05, 0.05, 0.5, 0.01]
    gammas = [5, 5, 1, 0.1, 1, 0.1, 1, 10]
    lrs=[0.001, 0.001, 0.001, 0.1, 0.001, 0.001, 0.001, 0.001]
    round_seps = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    n_iters = [20, 20, 10, 100, 30, 20, 20, 20]
    averaged = 5

    for ii, data_name in enumerate(data_names):
        for epsilon in [None, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.1, 0.01]:
            for kk in range(averaged):

                inputs, outputs = load_data("%s"%data_name)
                outputs = outputs.flatten()
                if 'shuttle' in data_name:
                    inputs = inputs.todense()

                train_input, test_input, train_output, test_output = train_test_split(inputs, outputs, test_size=0.2, shuffle=True) 

                mm = msc()#minmax scaler 

                input_mm = mm.fit_transform(train_input)
                test_input_mm = mm.transform(test_input)

                C = Cs[ii]
                gamma = gammas[ii]
                lr = lrs[ii]
                round_sep = round_seps[ii]
                n_iter = n_iters[ii]
                if epsilon is not None:
                    epsilon_SVDD = float(epsilon / 2)
                else:
                    epsilon_SVDD = epsilon

                start_time=time.time()
                clf = SVDD('rbf',C=C, gamma=gamma, lr=lr,fun=support_rkhs,epsilon=epsilon_SVDD,
                           sep_grad=True,round_sep=round_sep,n_iter=n_iter,nearest=1)  
                clf.fit(input_mm, train_output,200)

                intermediate_time = time.time()
                training_time = np.round(intermediate_time - start_time, 4)

                pred = clf.predict(test_input_mm)
                pred += np.min(train_output) - np.min(pred)
                acc = sum(pred==test_output)/len(test_output)
                test_time = np.round(time.time() - intermediate_time, 4)
                acc = np.round(acc.astype(float),4)
                print (data_name, "acc: ,", acc, "epsilon: ", epsilon, "C: ",C,"gamma: ", gamma,
                       "lr",lr,"round_sep",round_sep,"n_iter",n_iter,"training time: ",training_time, "test time: ", test_time)
                acc_list.append([data_name,acc, epsilon, C, gamma,lr,training_time, test_time])
                accs.append(acc)
            mean = np.round(np.mean(np.array(accs[-averaged:])),4)
            std = np.round(np.std(np.array(accs[-averaged:])),4)
            mean_stds.append([data_name,mean,std, epsilon, C, gamma,lr])
    print(mean_stds)


if __name__ == "__main__":
    main()

