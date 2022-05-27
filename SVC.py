"""
Python implementataion for Multi-Basin Support-Based Clustering Toolbox for Matlab 'https://github.com/SLCFLAB/Toolbox' (developed by D. Lee, S. Lee, K.-H. Jung, K. Kim, Y. Son, and Jaewook Lee) 

Developed by members of Statistical Learning and Computational Finance Laboratory (SLCF Lab), Department of Industrial Engineering, Seoul National University, Republic of Korea.
Refer https://github.com/SLCFLAB/Toolbox for the usage and the copyright.

"""

import numpy as np
import time
import scipy.io as sio
import sklearn.metrics.pairwise as kernels
import numpy.linalg as la
from scipy.linalg import inv
from scipy.sparse import csgraph as cg
from scipy.special import comb
from IPython import embed
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics.pairwise import euclidean_distances
from numpy import matlib as ml
from scipy.optimize import *
from sklearn.neighbors import NearestNeighbors as NNs
from sklearn.metrics import pairwise_distances
import pickle
import copy

##Structure.
"""
kradius
kernel
qpssvm
diagker
var_gpr
load_data
class support model
    __init__
    gp_normalize
    gp
    get_inv_C
    svdd_normalize
    svdd
class labeling
    __init__
    run
    findAdjMatrix
    cgsc
    findSEPs
    smsc(Byun)
    findTPs
    hierarchicalLabelTSVC
    tmsc
    fmsc
    vmsc
my_R1 // f for minimize
my_R2 // f, g, H
my_R_GP1 // f for minimize
my_R_GP2 // f, g, H
fsolve_R
fsolve_R_GP
"""
def cal_ARI(c1,c2):
    """
    calculates Rand Indices to compare two partitions
    type for c1 and c2 is numpy array tuple or list, i.e. one dimensional shape
    """
    c1 = np.array(c1)
    c2 = np.array(c2)
    C = contingency(c1,c2) #obtain contingency matrix
    n = np.sum(np.sum(C))
    nis = np.sum(np.square(np.sum(C,axis=1)))
    njs = np.sum(np.square(np.sum(C,axis=0)))
    #t1 =comb(n,2) 
    t1 = (n*(n-1))/2
    t2 = np.sum(np.square(C))
    t3 = 0.5*(nis+njs)

    nc = (n*(np.square(n)+1)-(n+1)*(nis+njs)+2*(nis*njs)/n)/(2*(n-1))
    A = t1+t2-t3
    D = -t2+t3
    if t1 ==nc:
        AR=0
    else:
        AR = (A-nc)/(t1-nc) # AR is ARI!
    RI = A/t1 #Rand 1971: Probability of agreement
    MI = D/t1 #Mirkin 1970: p(disagreement)
    HI = (A-D)/t1 #Hubert 1977: p(agree)-p(disagree)

    return AR

def contingency(c1,c2):
    """
    Calculate contingency matrix!
    First, we need to remap c1 and c2 as from zero to cluster number -1.
    """
    cc1 = np.unique(c1)
    cc2 = np.unique(c2)
    cd1 = {}
    cd2 = {}
    for i,c in enumerate(cc1):
        cd1[c] = i
    for i,c in enumerate(cc2):
        cd2[c] = i
    cp1 =np.array([cd1[ii] for ii in c1])
    cp2 =np.array([cd2[ii] for ii in c2])

    me1 = len(cc1)
    me2 = len(cc2)
    Cont = np.zeros((me1,me2))
    for i in range(len(c1)):
        Cont[cp1[i],cp2[i]] += 1

    return Cont
##############################################################
#place to change
#add Gaussain or Laplace noise to the grad 
def gradMinSVDD(x0, model, lr, iter,laplace=0, gaussian=0):
    momentum = 0 #0.3
#    nesterov = 0.1
    
    grad_list=[]
    if laplace ==0 and gaussian ==0:
        x = copy.deepcopy(x0)
    elif laplace!=0 and gaussian !=0:
        print("error!")
    elif laplace !=0:
        x = copy.deepcopy(x0) + np.random.laplace(0,laplace,1)
    else:
        x = copy.deepcopy(x0) + np.random.normal(0,gaussian)
    grads = [0,0]
    for i in range(iter):
        grad = fsolve_R(x, model)
#        grad = fsolve_R(x+nesterov*grads[0], model)
        grad_list.append(grad)
        grads[1]=grad
#         x -= lr*(grad)
        x -= lr*(momentum*grads[0] + grad)
#        x -= lr*(nesterov*grads[0] + grad)
        grads[0]=grads[1]
        
    return x, my_R1(x, model).flatten()[0],grad_list

def gradMinGP(x0, model, lr, iter,laplace=0, gaussian=0):
    momentum = 0.3
#    nesterov = 0.1
    
    grad_list=[]
    if laplace ==0 and gaussian ==0:
        x = copy.deepcopy(x0)
    elif laplace!=0 and gaussian !=0:
        print("error!")
    elif laplace !=0:
        x = copy.deepcopy(x0) + np.random.laplace(0,laplace,1)
    else:
        x = copy.deepcopy(x0) + np.random.normal(0,gaussian)
       
    grads = [0,0]
    for i in range(iter):
        grad = fsolve_R(x, model)
#        grad = fsolve_R(x-nesterov*grads[0], model)
        grad_list.append(grad)
        grads[1]=grad
#         x -= lr*(grad)
        x -= lr*(momentum*grads[0] + grad)
#        x -= lr*(nesterov*grads[0] + grad)
        grads[0]=grads[1]
    return x, my_R_GP1(x, model).flatten()[0], grad_list
################################################################
def kradius(X, model):

    # % KRADIUS computes the squared distance between vector in kernel space
    # % and the center of support.

    d = np.zeros([X.shape[1], 1])  ######################
    if model.support == 'SVDD':
        [dim, num_data] = X.shape
        x2 = diagker(X, model.svdd_params['ker'], model.svdd_params['arg'])
        Ksvx = kernel(input1=X, input2=model.svdd_model['sv']['X'], ker=model.svdd_params['ker'],
                      arg=model.svdd_params['arg'])
        d = x2 - 2 * np.dot(Ksvx, model.svdd_model['Alpha']) + model.svdd_model['b'] * np.ones((num_data, 1))
            
        if model.laplace_output != 0:
            d += np.random.laplace(0, scale=model.laplace_output, size=(num_data,1))
    elif model.support == 'GP':
        for i in range(X.shape[1]):
            # %[predict_label, accuracy] = var_gpr(X(:,i)', model.X, model.inv_C, model.hyperparams);
            predict_label = var_gpr(np.reshape(X[:, i], [-1, 1]), model.normalized_input, model.inv_C, model.gp_params)
            d[i][:] = predict_label.T
    elif model.support == 'pseudo':
        [dim, num_data] = X.shape
        x2 = diagker(X, model.pseudo_params['ker'], model.pseudo_params['arg'])
        Ksvx = kernel(input1=X, input2=model.pseudo_model['sv']['X'], ker=model.pseudo_params['ker'],
                      arg=model.pseudo_params['arg'])
        d = x2 - 2 * np.dot(Ksvx, model.pseudo_model['Alpha']) + model.pseudo_model['b'] * np.ones((num_data, 1))
        
    return d

def kernel(input1, ker, arg, input2=None):
    if input2 is None:

        input1 = input1.T
        if ker == 'linear':
            K = kernels.linear_kernel(input1)
            #   polynomial은 미구현. 파라미터가 좀 다른듯...
            #    if ker == 'poly':
            #        K = kernels.polynomial_kernel(input, )
        if ker == 'poly':
            K = kernels.polynomial_kernel(input1, degree=3, coef0=1)
        if ker == 'rbf':
            gamma = (0.5 / (arg * arg))
            K = kernels.rbf_kernel(input1, gamma=gamma)
        if ker == 'sigmoid':
            K = kernels.sigmoid_kernel(input1, gamma=arg[0], coef0=arg[1])
        return K

    else:
        input1 = input1.T
        input2 = input2.T
        if ker == 'linear':
            K = kernels.linear_kernel(input1, input2)
            #   polynomial은 미구현. 파라미터가 좀 다른듯...
            #    if ker == 'poly':
            #        K = kernels.polynomial_kernel(input, )
        if ker == 'poly':
            K = kernels.polynomial_kernel(input1,input2, degree=3, coef0=1)
        if ker == 'rbf':
            gamma = (0.5 / (arg * arg))
            K = kernels.rbf_kernel(input1, input2, gamma=gamma)
        if ker == 'sigmoid':
            K = kernels.sigmoid_kernel(input1, input2, gamma=arg[0], coef0=arg[1])

        return K

def qpssvm(H, f, b, I): #laplace_fm = lambda
    H = H.astype(np.double)

    P = matrix(H, tc='d')
    f = f.astype(np.double)
    

    q = matrix(f.reshape(f.shape[0], 1), tc='d')

    G1 = np.eye(I.shape[0])
    G2 = np.eye(I.shape[0])
    G2 = (-1) * G2
    G = matrix(np.concatenate((G1, G2), axis=0), tc='d')

    h1 = np.repeat(b, I.shape[0])
    h1 = h1.reshape(h1.shape[0], 1)
    h2 = np.repeat(0, I.shape[0])
    h2 = h2.reshape(h2.shape[0], 1)
    h = matrix(np.concatenate((h1, h2), axis=0), tc='d')

    A1 = np.ones(I.shape[0]).reshape(1,-1)
    A = matrix(A1, tc= 'd')
    b = matrix(1, tc='d')
    
    
    sol = solvers.qp(P, q, G, h, A, b)

    x = sol['x']
    x = np.array(x)
    fval = sol['primal objective']
    return [x, fval]

def diagker(X, ker, arg):
    diagK = np.diag(kernel(input1=X, ker=ker, arg=arg))
    diagK = diagK.reshape(diagK.shape[0], 1)
    return diagK

def var_gpr(test, input, inv_C, hyperpara):
    # % Variance in Gaussian Process Regression used as Support Funtion of Clustering
    # %
    # % The variance function of a predictive distribution of GPR
    # % sigma^2(x) = kappa - k'C^(-1)k

    [D, n] = input.shape
    [D, nn] = test.shape
    expX = hyperpara
    a = np.zeros([nn, n])

    for d in range(D):
        a = a + expX[0][d] * (np.tile(input[d, :], [nn, 1]) - np.tile(np.reshape(test[d, :], [-1, 1]), [1, n])) ** 2

    a = expX[1] * np.exp(-0.5 * a)
    b = expX[1]
    mul = a.dot(inv_C.T)
    dmul = np.multiply(a, mul)
    s_a_inv_C_a = np.sum(dmul, axis=1)
    var = b - s_a_inv_C_a

    return var

def load_data(data_name):
    if 'three_moons' in data_name:
        input, output= make_moons(n_samples=400, noise=0.1, random_state=0)
        input = input.T
    else:
        data=sio.loadmat('./dataset/'+data_name)
        if 'toy' in data_name:
            input=data['X']
        elif 'ring' in data_name:
            input=data['input'].T
            output=data['output']
        else:
            input=data['train_input']
            output=data['train_output']
    return input, output

def svdd_normalize(input_data,svdd_model):
    Xin = svdd_model.input
    [dim, n] = input_data.shape
    mean_by_col = np.mean(Xin, axis=1).reshape(dim, 1)
    stds_by_col = np.std(Xin, axis=1).reshape(dim, 1)
    means = np.tile(mean_by_col, (1, n))
    stds = np.tile(stds_by_col, (1, n))
    X_normal = (input_data - means) / stds
    return X_normal 
def svdd_denormalize(input_data,svdd_model):
    Xin = svdd_model.input
    [dim,n] = input_data.shape
    mean_by_col = np.mean(Xin, axis=1).reshape(dim, 1)
    stds_by_col = np.std(Xin, axis=1).reshape(dim, 1)
    means = np.tile(mean_by_col, (1, n))
    stds = np.tile(stds_by_col, (1, n))
    X_denormal = (input_data * stds)+means
    return X_denormal

import numbers
import array
from collections.abc import Iterable

import numpy as np
from scipy import linalg
import scipy.sparse as sp


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
# from sklearn.utils.validation import _deprecate_positional_args

def make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None):
    """Make two interleaving half circles.
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each of two moons.
        .. versionchanged:: 0.23
           Added two-element tuple.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """

    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 3
        n_samples_in = n_samples // 3
        n_samples_new = n_samples - 2*n_samples_out
        
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError('`n_samples` can be either an int or '
                             'a two-element tuple.') from e

    generator = check_random_state(random_state)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5
    new_circ_x = np.cos(np.linspace(0, np.pi, n_samples_new)) + 2
    new_circ_y = np.sin(np.linspace(0, np.pi, n_samples_new))     

    X = np.vstack([np.append(np.append(outer_circ_x, inner_circ_x),new_circ_x),
                   np.append(np.append(outer_circ_y, inner_circ_y), new_circ_y)]).T

    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp), 2*np.ones(n_samples_new, dtype = np.intp)])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X.T, y.astype(int)


class supportmodel:
    def __init__(self, input=None, support='SVDD', hyperparams=None,laplace_grad = 0,laplace_output = 0, laplace_alpha=0, laplace_alpha_bf=0, laplace_fm = 0, gaussian=0):
        self.radius = None
        self.ind = None
        self.input = input  ### input.shape = [dim, N_sample]
        self.laplace=laplace_grad
        self.laplace_output = laplace_output
        self.laplace_alpha = laplace_alpha
        self.laplace_alpha_bf = laplace_alpha_bf
        self.laplace_fm= laplace_fm

        self.gaussian=gaussian
        if input is not None:
            assert type(input) is np.ndarray, 'ERROR: input type must be numpy.ndarray'
        self.support = support  ### 'SVDD' or 'GP'
        assert self.support == 'SVDD' or self.support == 'GP' or self.support == 'pseudo', 'ERROR: Support must be \'SVDD\', or \'GP\', or \'psuedo\''
        if input is not None:
            if self.support == 'SVDD':
                if hyperparams == None:
                    hyperparams = {'ker': 'rbf', 'arg': 1, 'solver': 'imdm', 'C': 1}
#                 self.svdd_normalize()
                self.normalized_input = self.input
                self.svdd_params = hyperparams
                self.svdd()
            elif self.support == 'GP':
                if hyperparams == None:
                    hyperparams = [100*np.ones((input.shape[0],1)), 1, 10]
#                 self.gp_normalize()
                self.normalized_input = self.input
                self.gp_params = hyperparams
                assert self.gp_params[0].shape[0] == self.input.shape[0], "ERROR: invalid gp_params shape"
                self.gp()
            elif self.support == 'pseudo':
                if hyperparams == None:
                    hyperparams = {'ker': 'rbf', 'arg': 1, 'solver': 'imdm', 'C':1} #####lsy: other params??
                self.normalized_input = self.input
#                 self.svdd_normalize() #####lsy: pseudo_normalize()??
                self.pseudo_params = hyperparams
                self.pseudo()
    def save(self,save_path = None):
        assert save_path is not None
        f = open(save_path,'wb')
        pickle.dump(self,f)
        f.close()
        self.save_path = save_path

    def load(self,load_path = None):
        assert load_path is not None
        f = open(load_path,'rb')
        self = pickle.load(f)
        f.close()
        self.load_path = load_path

    def gp_normalize(self):  ##my_normalize2
        # % input [dim x num_data]
        # % output [dim x num_data]

        [dim, num] = self.input.shape
        max_val = np.max(np.max(self.input))  ###############np
        min_val = np.min(np.min(self.input))
        self.normalized_input = (self.input - np.tile(min_val, [dim, num])) / np.tile(max_val - min_val, [dim, num])

    def gp(self):  ###using var_gpr
        ## Gaussian Process Support Function for Clustering
        ##
        ##  Gaussian process support function which is the variance function of a
        ##  predictive distribution of GPR :
        ##   sigma^2(x) = kappa - k'C^(-1)k
        ##  where covariance matrix C(i,j) is a parameterized function of x(i) and
        ##  x(j) with hyperparameters Theta, C(i,j) = C(x(i),x(j);Theta),
        ##  kappa = C(x~,x~;Theta) for a new data point x~ = x(n+1),
        ##  k = [C(x~,x1;Theta),...,C(x~,x(n);Theta)]
        ##
        ## Synopsis:
        ##  model = gp(input)
        ##  model = gp(input,hyperparams)
        ##
        ## Description :
        ## It computes variance function of gaussian process regression learned from
        ## a training data which can be an estimate of the support of a probability
        ## density function. A dynamic process associated with the variance function
        ## can be built and applied to cluster labeling of the data points. The
        ## variance function estimates the support region by sigma^2 <= theta, where
        ## theta = max(sigma^2(x))
        ##
        ## Input:
        ##  input [dim x num_data] Input data.
        ##  hyperparams [(num_data + 2) x 1]
        ##
        ## Output:
        ##  model [struct] Center of the ball in the kernel feature space:
        ##   .input [dim x num_data]
        ##   .X [num_data x dim]
        ##   .hyperparams [(num_data + 2) x 1]
        ##   .inside_ind [1 x num_data] : all input points are required to be used for computing
        ##   support function value.
        ##   .inv_C [num_data x num_data] : inverse of a covariance matrix C for the training inputs
        ##   .r : support function level with which covers estimated support region

        print("-------------------------------------------")
        print('Step 1 : Training Support Function by GP...')
        start_time = time.time()

        self.inside_ind = range(self.normalized_input.shape[1])
        self.get_inv_C()
        tmp = var_gpr(self.normalized_input, self.normalized_input, self.inv_C, self.gp_params)  #######self
        self.R = max(tmp)

        self.training_time = time.time() - start_time
        print("---------------------")
        print('Training Completed !!!!')
        print("Training time for GP is : ", self.training_time, " sec")

    def get_inv_C(self):
        ## Compute inverse of a covariance matrix C

        ninput = self.normalized_input
        params = self.gp_params
        [D, n] = ninput.shape  ## dimension of input space and number of training cases
        C = np.zeros([n, n])

        for d in range(D):
            C = C + params[0][d] * (np.tile(np.reshape(ninput[d, :], [-1, 1]), [1, n]) - np.tile(ninput[d, :],
                                                                                                 [n, 1])) ** 2
        C = params[1] * np.exp(-0.5 * C) + params[2] * np.eye(n)

        self.inv_C = inv(np.matrix(C))

    def svdd_normalize(self): ### turn off nomalize now
        Xin = self.input
        [dim, n] = Xin.shape

        mean_by_col = np.mean(Xin, axis=1).reshape(dim, 1)
        stds_by_col = np.std(Xin, axis=1).reshape(dim, 1)

        means = np.tile(mean_by_col, (1, n))
        stds = np.tile(stds_by_col, (1, n))

        X_normal = (Xin - means) / stds

        self.normalized_input = X_normal#X_normal
        # print("Normalized input : ", self.normalized_input)

    def svdd(self):
        print('---------------------------------------------')
        print("Step 1: Training Support Function by SVDD ...")
        start_time = time.time()

        model = {}
        # self.model['support'] = 'SVDD'
        options = self.svdd_params
        if options.get('ker') == None:
            options['ker'] = 'rbf'
            print("You need to specify kernel type with \'ker\'")
        if options.get('arg') == None:
            options['arg'] = 1
            print("You need to specify kernel parameters with \'arg\', consistently with kernel type")
        if options.get('solver') == None:
            options['solver'] = 'imdm'
            print("You must specify solver type with \'solver\'")
        if options.get('C') == None:
            options['C'] = 1
            print("You must specifly svm parameter C with \'C\'")

        [dim, num_data] = self.normalized_input.shape

        # Set up QP Problem
        K = kernel(input1=self.normalized_input, ker=options['ker'], arg=options['arg'])
        f = -np.diag(K)
        H = 2 * K
        b = options['C']
        I = np.arange(num_data)
        if self.laplace_fm is not 0:
            [Alpha, fval] = qpssvm(H, f, b, I, laplace_fm = self.laplace_fm)  # 아직 qpssvm에서 stat 미구현
        else:
            [Alpha, fval] = qpssvm(H, f, b, I)
        # print(Alpha)
        # inx = np.where(Alpha > pow(10, -5))[0]  # Alpha를 0이상으로 잡으면 잘못잡힘.
        # self.model['support_type'] = 'SVDD'
        
        if self.laplace_alpha_bf is not 0:
#             Alpha[inx] += np.random.laplace(0, scale=self.laplace_alpha, size=(len(Alpha[inx]),1))
            Alpha+= np.random.laplace(0, scale=self.laplace_alpha_bf, size=(len(Alpha),1))
            Alpha = np.clip(Alpha, 0, options['C'])
        
        inx = (Alpha > pow(10, -5)).ravel()  
        
        if self.laplace_alpha is not 0:
            Alpha[inx] += np.random.laplace(0, scale=self.laplace_alpha, size=(len(Alpha[inx]),1))

        model['Alpha'] = Alpha[inx]     
        model['sv_ind'] = np.logical_and(Alpha > pow(10, -5), Alpha < (options['C'] - pow(10, -7))).ravel()
        # print(self.model['sv_ind'])
        model['bsv_ind'] = (Alpha >= (options['C'] - pow(10, -7))).ravel()
        model['inside_ind'] = (Alpha < (options['C'] - pow(10, -7))).ravel()
        # print(Alpha[inx].shape, K[inx,:][:,inx].shape)
        model['b'] = np.dot(np.dot(Alpha[inx].T, K[inx, :][:, inx]), Alpha[inx])

        # setup model
        model['sv'] = {}
        model['sv']['X'] = self.normalized_input[:, inx]

        model['sv']['inx'] = inx

        model['nsv'] = np.count_nonzero(inx)    ## inx has elements - True or False

        model['options'] = options
        #    model['stat'] = stat
        model['fun'] = 'kradius'
        self.svdd_model = model
        self.n = num_data
                
        radius = kradius(self.normalized_input[:, model['sv_ind']], self)
        self.ind = self.normalized_input[:, model['sv_ind']]
        self.radius = radius
        self.R = np.amax(radius)
        print('-------------------')
        print("Training Completed!!!!")
        self.training_time = time.time() - start_time
        print("Traning time for SVDD is : ", self.training_time, " sec")

    def pseudo(self):
        print('----------------------------------------------------------------')
        print("Step 1: Training Support Function by pseudo density function ...")
        start_time = time.time()

        model = {}
        # self.model['support'] = 'SVDD'
        options = self.pseudo_params
        if options.get('ker') == None:
            options['ker'] = 'rbf'
            print("You need to specify kernel type with \'ker\'")
        if options.get('arg') == None:
            options['arg'] = 1
            print("You need to specify kernel parameters with \'arg\', consistently with kernel type")
        if options.get('solver') == None:
            options['solver'] = 'imdm'
            print("You must specify solver type with \'solver\'")
        if options.get('C') == None:
            options['C'] = 1
            print("You must specifly svm parameter C with \'C\'")

        [dim, num_data] = self.normalized_input.shape

        # Set up QP Problem
        K = kernel(input1=self.normalized_input, ker=options['ker'], arg=options['arg'])
        f = -np.diag(K)
        H = 2 * K
        b = options['C']
        I = np.arange(num_data)
        [Alpha, fval] = qpssvm(H, f, b, I)  # 아직 qpssvm에서 stat 미구현
        # print(Alpha)
        # inx = np.where(Alpha > pow(10, -5))[0]  # Alpha를 0이상으로 잡으면 잘못잡힘.
        inx = (Alpha > pow(10, -5)).ravel()
        # self.model['support_type'] = 'SVDD'

        model['Alpha'] = Alpha[inx]
        model['sv_ind'] = np.logical_and(Alpha > pow(10, -5), Alpha < (options['C'] - pow(10, -7))).ravel()
        # print(self.model['sv_ind'])
        model['bsv_ind'] = (Alpha >= (options['C'] - pow(10, -7))).ravel()
        model['inside_ind'] = (Alpha < (options['C'] - pow(10, -7))).ravel()
        # print(Alpha[inx].shape, K[inx,:][:,inx].shape)
        model['b'] = np.dot(np.dot(Alpha[inx].T, K[inx, :][:, inx]), Alpha[inx])

        # setup model
        model['sv'] = {}
        model['sv']['X'] = self.normalized_input[:, inx]

        model['sv']['inx'] = inx

        model['nsv'] = np.count_nonzero(inx)    ## inx has elements - True or False

        model['options'] = options
        #    model['stat'] = stat
        model['fun'] = 'kradius'
        self.pseudo_model = model

        radius = kradius(self.normalized_input[:, model['sv_ind']], self)
        self.R = np.amax(radius)
        print('--------------------')
        print("Training Completed!")
        self.training_time = time.time() - start_time
        print("Training time for pseudo density is : ", self.training_time, " sec")


class labeling:
    def __init__(self, target = None, supportmodel=None, labelingmethod='CG-SC', options=None, laplace=0, gaussian=0):
        self.target = target
        self.supportmodel = supportmodel
        self.labelingmethod = labelingmethod
        self.options = options
        self.locals=locals
        self.laplace=laplace
        self.gaussian=gaussian
        if supportmodel is not None:
            print('----------------------------------------------')
            print("Step 2 : Labeling by the method " + self.labelingmethod + "...")
            start_time = time.time()
            self.run()
            self.labeling_time = time.time() - start_time
            print('-------------------')
            print("Labeling Completed!")
            print("Time for labeling is : ", self.labeling_time, " sec!")
            print("-------------------------------------------------")
    def save(self,save_path = None):
        assert save_path is not None
        f = open(save_path,'wb')
        pickle.dump(self,f)
        f.close()
        self.save_path = save_path
    def load(self,load_path = None):
        assert load_path is not None
        f = open(load_path,'rb')
        self = pickle.load(f)
        f.close()
        self.load_path = load_path

    def run(self):
        if self.labelingmethod == 'CG-SC':
            self.cgsc()
        elif self.labelingmethod == 'S-MSC':
            self.smsc()
        elif self.labelingmethod == 'T-MSC':
            self.tmsc()
        elif self.labelingmethod == 'F-MSC':
            self.fmsc()
        elif self.labelingmethod == 'V-MSC':
            self.vmsc()
        else:
            print("ERROR: invalid labeling method; valid examples (CG-SC, S-MSC, T-MSC, F-MSC, V-MSC)")

    def findAdjMatrix(self, input):
        model = self.supportmodel
        N = input.shape[1]
        start_time = time.time()
        adjacent = np.zeros([N, N])
        if type(model)==supportmodel:
            R = model.R + 10 ** (-7)  # % Squared radius of the minimal enclosing ball
            self.lineseg=[]
            for i in range(N):  ##rows
                for j in range(N):  ##columns
                    ## if the j is adjacent to i - then all j adjacent's are also adjacent to i.
                    if j == i:
                        adjacent[i, j] = 1
                    elif j < i:
                        if (adjacent[i, j] == 1):
                            adjacent[i, :] = np.logical_or(adjacent[i, :], adjacent[j, :])
                            adjacent[:, i] = adjacent[i, :]
                    else:
                        ## if adajecancy already found - no point in checking again
                        if (adjacent[i, j] != 1):
                            ## goes over 10 points in the interval between these 2 Sample points
                            adj_flag = 1  ## unless a point on the path exits the shpere - the points are adjacnet
                            for interval in [0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1]:
                                z = input[:, i] + interval * (input[:, j] - input[:, i])
                                z = np.reshape(z, [-1, 1])
                                ## calculates the sub-point distance from the sphere's center
                                d = kradius(z, model)
                                if d > R:
                                    adj_flag = 0
                                    self.lineseg.append(interval)
                                    break
                            if adj_flag == 1:
                                adjacent[i, j] = 1
                                adjacent[j, i] = 1
        else:
            R = 10 ** (-7)  # % Squared radius of the minimal enclosing ball
            self.lineseg=[]
            for i in range(N):  ##rows
                for j in range(N):  ##columns
                    ## if the j is adjacent to i - then all j adjacent's are also adjacent to i.
                    if j == i:
                        adjacent[i, j] = 1
                    elif j < i:
                        if (adjacent[i, j] == 1):
                            adjacent[i, :] = np.logical_or(adjacent[i, :], adjacent[j, :])
                            adjacent[:, i] = adjacent[i, :]
                    else:
                        ## if adajecancy already found - no point in checking again
                        if (adjacent[i, j] != 1):
                            ## goes over 10 points in the interval between these 2 Sample points
                            adj_flag = 1  ## unless a point on the path exits the shpere - the points are adjacnet
                            for interval in [0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1]:
                                z = input[:, i] + interval * (input[:, j] - input[:, i])
                                z = np.reshape(z, [-1, 1])
                                ## calculates the sub-point distance from the sphere's center
                                d = my_R1_cond(z, model)
                                if d > R:
                                    adj_flag = 0
                                    self.lineseg.append(interval)
                                    break
                            if adj_flag == 1:
                                adjacent[i, j] = 1
                                adjacent[j, i] = 1

        self.adjacent_matrix = adjacent
        self.symmetric = (np.max(np.abs(adjacent - adjacent.T) == 0))
        print("time to find adjacent = ", time.time() - start_time)

    def cgsc(self):
        # % CGSVC Support Vector Clusteing using Complete-Graph Based Labeling Method
        # %
        # % Description:
        # %  To determine whether a pair of xi and xj is in the same contour,
        # %  it can be used a complete-graph(CG) strategy that relies on the fact
        # %  that any path connecting two data points in different contours must
        # %  exit the contours in data space, which is equivalent that the image
        # %  of the path in feature space exits the minimum enclosing sphere.
        # %  CG strategy makes adjacency matrix Aij between pairs of points
        # %  xi and xj as follows :
        # %   A(ij) = 1 if for all y on the line segment connecting xi and xj
        # %           0 otherwise

        model = self.supportmodel
        self.findAdjMatrix(model.normalized_input)
        self.cluster_label = cg.connected_components(self.adjacent_matrix)  # [1], cluster_lables????????
        print(self.cluster_label[1])

    def findSEPs(self):
        model = self.supportmodel
        X = model.normalized_input

        [dim, N] = X.shape
        N_locals = []
        local_val = []

        if model.support == 'GP':
            for i in range(N):
                x0 = X[:, i]
                if self.options['grad']:
                    temp,val, grad_list = gradMinGP(x0, model, self.options['lr'], self.options['iter'],laplace=self.laplace, gaussian=self.gaussian)
                else:
                    res = minimize(fun=my_R_GP1, x0=x0, args=model, method="Nelder-Mead")
                    [temp, val] = [res.x, res.fun]
                    grad_list=['grad:False']
               
                N_locals.append(temp)
                local_val.append(val)
                

        elif model.support == 'SVDD':
            for i in range(N):
                x0 = X[:, i]
                if self.options['grad']:
                    temp,val,grad_list = gradMinSVDD(x0, model, self.options['lr'], self.options['iter'],laplace=self.laplace, gaussian=self.gaussian)
                else:
                    if len(x0) <= 2:
                        res = minimize(fun=my_R1, x0=x0, args=model, method='Nelder-Mead')
                        [temp, val] = [res.x, res.fun]
                    else:
                        res = minimize(fun=my_R1, x0=x0, args=model, method='trust-ncg',jac = my_R_grad,hess = my_R_hess)
                        [temp, val] = [res.x, res.fun]
                    grad_list=['grad:False']
                N_locals.append(temp)
                local_val.append(val)
        
        elif model.support == 'pseudo':
            for i in range(N):
                x0 = X[:, i]
                if len(x0) <= 2:
                    res = minimize(fun=my_R_pseudo, x0=x0, args=model, method='Nelder-Mead')
                    [temp, val] = [res.x, res.fun]
                else:
                    res = minimize(fun=my_R_pseudo, x0=x0, args=model, method='trust-ncg',jac = my_R_pseudo_grad, hess = my_R_pseudo_Hess)
                    [temp, val] = [res.x, res.fun]
                N_locals.append(temp)
                local_val.append(val)
                
        N_locals = np.array(N_locals)
        self.N_locals = N_locals
        local_val = np.array(local_val)
        
        local, I, match_local = np.unique(np.round(10 * N_locals), axis=0, return_index=True, return_inverse=True)
        self.match_local = match_local
        newlocal = N_locals[I, :]
        newlocal_val = local_val[I]
     
        return [N_locals, newlocal, newlocal_val, match_local], grad_list

    def findSEPs_cond(self): ##only for SVDD, not for grad yet
        models = self.supportmodel
        
        X = models[0].normalized_input
        for i in range(len(models)-1):
            X = np.hstack((X, models[i+1].normalized_input))

        [dim, N] = X.shape
        N_locals = []
        local_val = []               

        if models[0].support == 'SVDD':
            for i in range(N):
                x0 = X[:, i]               
                if len(x0) <= 2:
                    res = minimize(fun=my_R1_cond, x0=x0, args=models, method='Nelder-Mead')
                    [temp, val] = [res.x, res.fun]
                else:
                    res = minimize(fun=my_R1_cond, x0=x0, args=models, method='trust-ncg')
                    [temp, val] = [res.x, res.fun]
                grad_list=['grad:False']
                N_locals.append(temp)
                local_val.append(val)
                
        N_locals = np.array(N_locals)
        self.N_locals = N_locals
        local_val = np.array(local_val)
        
        local, I, match_local = np.unique(np.round(10 * N_locals), axis=0, return_index=True, return_inverse=True)
        self.match_local = match_local
        newlocal = N_locals[I, :]
        newlocal_val = local_val[I]
     
        return [N_locals, newlocal, newlocal_val, match_local], grad_list

    def smsc(self):
        t1 = time.time()
        if type(self.supportmodel)==supportmodel:
            [rep_locals, locals, local_val, match_local], grad_list = self.findSEPs()
        else:
            [rep_locals, locals, local_val, match_local], grad_list = self.findSEPs_cond()
        print("time to find SEPs = ", time.time() - t1)
        if self.target is not None:
            sep_label = []
            for i in np.unique(self.match_local):
                sep_label.append(np.argmax(np.bincount(self.target[self.match_local==i])))
            self.local_label = sep_label
        # %% Step 2 : Labeling Data for Clustering
        self.locals = locals
        self.findAdjMatrix(locals.T)
        self.grad_list=grad_list
        #print(self.grad_list)
        # Finds the cluster assignment of each data point
        # clusters = findConnectedComponents(self.adjacent_matrix)
        csm = cg.connected_components(self.adjacent_matrix)
        local_cluster_assignments = csm[1]
        local_cluster_assignments = np.array(local_cluster_assignments)
        self.local_cluster = local_cluster_assignments
#         print(local_cluster_assignments)
        self.cluster_label = local_cluster_assignments[match_local]
#         print(self.cluster_label)
    def inference_smsc(self, X_test, inference_voronoi = False):
        if self.target is not None:
            model = self.supportmodel
            [dim, N] = X_test.shape
            pred = []
            
            if inference_voronoi:
                for i in range(N):
                    x0 = X_test[:,i]
                    temp_dist = pairwise_distances([x0], self.locals)
                    pred.append(self.local_label[np.argmin(temp_dist[0])])

            else:
                if type(model)==supportmodel:
                    for i in range(N):
                        x0 = X_test[:,i]
                        if len(x0) <= 2:
                            res = minimize(fun=my_R1, x0=x0, args=model, method='Nelder-Mead')
                            [temp, val] = [res.x, res.fun]
                        else:
                            res = minimize(fun=my_R1, x0=x0, args=model, method='trust-ncg')
                            [temp, val] = [res.x, res.fun]
                        temp_dist = pairwise_distances([temp], self.locals)
                        print(temp_dist,np.argmin(temp_dist[0]))
                        pred.append(self.local_label[np.argmin(temp_dist[0])])                
                else:
                    for i in range(N):
                        x0 = X_test[:,i]
                        if len(x0) <= 2:
                            res = minimize(fun=my_R1_cond, x0=x0, args=model, method='Nelder-Mead')
                            [temp, val] = [res.x, res.fun]
                        else:
                            res = minimize(fun=my_R1_cond, x0=x0, args=model, method='trust-ncg')
                            [temp, val] = [res.x, res.fun]
                        temp_dist = pairwise_distances([temp], self.locals)
                        pred.append(self.local_label[np.argmin(temp_dist[0])])        
            return pred
        else:
            print("Cannot inference. No target!")

    def findTPs(self):
        locals = self.locals
        model = self.supportmodel
        epsilon = self.options['epsilon']
        R = model.R + 10 ** (-7)

        ts = {}
        ts['x'] = []
        ts['f'] = []
        ts['neighbor'] = []
        ts['purturb'] = []
        [N, attr] = locals.shape
        tmp_x = []

        if model.support == 'GP':
            for i in range(N):
                for j in range(i, N):
                    for k in range(10):
                        x0 = locals[i] + 0.1 * (k+1) * (locals[j] - locals[i])
                        sep = fsolve(func=fsolve_R_GP, x0=x0, args=model, xtol=10 ** (-6))
                        tmp_x.append(sep)
            tmp_x = np.array(tmp_x)
            [dummy, I, J] = np.unique(np.round(10*tmp_x),axis=0, return_index=True, return_inverse=True)
            tmp_x = tmp_x[I, :]
            for i in range(list(tmp_x.shape)[0]):
                sep = tmp_x[i]
                [f, g, H] = my_R_GP2(sep, model)
                [D, V] = la.eig(H)
                ind = []
                if np.sum(D < 0) == 1:
                    sep1 = sep + epsilon * V[np.where(D < 0)]
                    sep2 = sep - epsilon * V[np.where(D < 0)]
                    if attr == 2:
                        res1 = minimize(fun=my_R_GP1, x0=sep1, args=model, method='Nelder-Mead')
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_GP1, x0=sep2, args=model, method='Nelder-Mead')
                        [temp2, val] = [res2.x, res2.fun]
                    else:
                        res1 = minimize(fun=my_R_GP1, x0=sep1, args=model, hess=True)
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_GP1, x0=sep2, args=model, hess=True)
                        [temp2, val] = [res2.x, res2.fun]
                    [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp1.reshape(1, -1), locals))]
                    [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp2.reshape(1, -1), locals))]
                    if ind1 != ind2:
                        ts['x'].append(sep)
                        ts['f'].append(f)
                        ts['neighbor'].append([ind1, ind2])
                        ts['purturb'].append([sep1, sep2])

        if model.support == 'SVDD':
            for i in range(N):
                for j in range(i, N):
                    for k in range(10):
                        x0 = locals[i] + 0.1 * (k + 1) * (locals[j] - locals[i])
                        sep = fsolve(func=fsolve_R, x0=x0, args=model, maxfev=300, xtol=10 ** (-6))
                        tmp_x.append(sep)
            tmp_x = np.array(tmp_x)
            [dummy, I, J] = np.unique(np.round(10 * tmp_x), axis=0, return_index=True, return_inverse=True)
            tmp_x = tmp_x[I, :]
            for i in range(list(tmp_x.shape)[0]):
                sep = tmp_x[i]
                [f, g, H] = my_R2(sep, model)
                [D, V] = la.eig(H)
                ind = []
                if np.sum(D < 0) == 1:
                    sep1 = sep + epsilon * V[np.where(D < 0)]
                    sep2 = sep - epsilon * V[np.where(D < 0)]
                    if attr == 2:
                        res1 = minimize(fun=my_R1, x0=sep1, args=model, method='Nelder-Mead')
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R1, x0=sep2, args=model, method='Nelder-Mead')
                        [temp2, val] = [res2.x, res2.fun]
                    else:
                        res1 = minimize(fun=my_R1, x0=sep1, args=model,jac = my_R_grad, hess=my_R_hess)
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R1, x0=sep2, args=model,jac = my_R_grad, hess=my_R_hess)
                        [temp2, val] = [res2.x, res2.fun]
                    [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp1.reshape(1, -1), locals))]
                    [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp2.reshape(1, -1), locals))]
                    if ind1 != ind2:
                        ts['x'].append(sep)
                        ts['f'].append(f)
                        ts['neighbor'].append([ind1, ind2])
                        ts['purturb'].append([sep1, sep2])


        if model.support == 'pseudo':
            for i in range(N):
                for j in range(i, N):
                    for k in range(10):
                        x0 = locals[i] + 0.1 * (k + 1) * (locals[j] - locals[i])
                        sep = fsolve(func=fsolve_R_pseudo, x0=x0, args=model, maxfev=300, xtol=10 ** (-6)) #####lsy
                        tmp_x.append(sep)
            tmp_x = np.array(tmp_x)
            [dummy, I, J] = np.unique(np.round(10 * tmp_x), axis=0, return_index=True, return_inverse=True)
            tmp_x = tmp_x[I, :]
            print(tmp_x)
            for i in range(list(tmp_x.shape)[0]):
                sep = tmp_x[i]
                [f, g, H] = my_R2_pseudo(sep, model) ######lsy
                [D, V] = la.eig(H)
                ind = []
                if np.sum(D < 0) == 1:
                    sep1 = sep + epsilon * V[np.where(D < 0)]
                    sep2 = sep - epsilon * V[np.where(D < 0)]
                    if attr == 2:
                        res1 = minimize(fun=my_R_pseudo, x0=sep1, args=model, method='Nelder-Mead')
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_pseudo, x0=sep2, args=model, method='Nelder-Mead')
                        [temp2, val] = [res2.x, res2.fun]
                    else:
                        res1 = minimize(fun=my_R_pseudo, x0=sep1, args=model,method = 'trust-ncg',jac = my_R_pseudo_grad, hess=my_R_pseudo_Hess)
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_pseudo, x0=sep2, args=model,method = 'trust-ncg',jac = my_R_pseudo_grad, hess=my_R_pseudo_Hess)
                        [temp2, val] = [res2.x, res2.fun]
                    [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp1.reshape(1, -1), locals))]
                    [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp2.reshape(1, -1), locals))]
                    if ind1 != ind2:
                        ts['x'].append(sep)
                        ts['f'].append(f)
                        ts['neighbor'].append([ind1, ind2])
                        ts['purturb'].append([sep1, sep2])
                        
        ts['x'] = np.array(ts['x'])
        ts['f'] = np.array(ts['f'])
        ts['neighbor'] = np.array(ts['neighbor'])
        ts['purturb'] = np.array(ts['purturb'])
        self.ts = ts

    def hierarchicalLabelTSVC(self):
        print("hierarchicalLableTSVC")
        nOfLocals = self.locals.shape[0]
        ts = self.ts
        nOfTS = len(ts['f'])
        K = self.options['K']

        local_clusters_assignments = []
        f_sort = np.sort(ts['f'], 0)  # small --> large
        print("f_sort:", f_sort)
        adjacent = np.zeros([nOfLocals, nOfLocals, nOfTS])
        a = []
        flag = 0
        for m in range(nOfTS):
            cur_f = f_sort[
                -m - 1]  # % cutting level:large --> small  (small number of clusters --> large number of clusters)
            # %cur_f=f_sort(i);         % cutting level: small --> large (large number of clusters --> small number of clusters)

            tmp = np.nonzero(ts['f'] < cur_f)[0]
            if len(tmp) > 0:  # % TSs inside the sphere
                for j in range(len(tmp)):
                    adjacent[ts['neighbor'][tmp[j], 0], ts['neighbor'][tmp[j], 1], m] = 1
                    adjacent[ts['neighbor'][tmp[j], 1], ts['neighbor'][tmp[j], 0], m] = 1
                    # %% To connect nodes which can be connected via directly connected edges.
                for i in range(nOfLocals):
                    for j in range(i):
                        if (adjacent[i, j, m] == 1):
                            adjacent[i, :, m] = np.logical_or(adjacent[i, :, m], adjacent[j, :, m])
                    adjacent[i, i] = 1

            a = [a, cur_f]
            my_ts = {}
            my_ts['x'] = ts['x'][tmp, :]
            my_ts['f'] = ts['f'][tmp, :]
            my_ts['purturb'] = ts['purturb'][tmp, :]
            my_ts['neighbor'] = ts['neighbor'][tmp, :]
            my_ts['cuttingLevel'] = cur_f
            ind = np.nonzero(ts['f'] == cur_f)[0]
            my_ts['levelx'] = ts['x'][ind[0], :]
            tmp_ts = {}  ####dictionary
            tmp_ts[m] = my_ts

            assignment = cg.connected_components(adjacent[:, :, m])[1]
            print("assignment:", assignment)
            print("N_clusters:", np.max(assignment) + 1)
            if np.max(assignment) == K - 1:
                print('We can find the number of K clusters');
                # % clstmodel update
                self.out_ts = tmp_ts[m]
                # % cluster assignment into entire data points
                self.local_ass = assignment
                self.cluster_label = self.local_ass[self.match_local].T
                flag = 1
                break

            local_clusters_assignments = [local_clusters_assignments, assignment]

            # % cannot find k clusters
        if flag == 0:
            print(
                'Cannot find cluster assignments with K number of clusters, instead that we find cluster assignments the with the nearest number of clusters to K !');
            [dummy, ind] = np.min(euclidean_distances(np.max(local_clusters_assignments, 0).T, K), 0)  ####min/max

            # %ts=[];
            self.out_ts = tmp_ts[ind[0]]
            local_clusters_assignments = local_clusters_assignments[:, ind[0]]
            self.local_ass = local_clusters_assignments
            self.cluster_label = self.local_ass[self.match_local]
            print(self.cluster_label)

    def tmsc(self):
        fHierarchical = self.options['hierarchical']

        # % Find SEPs
        [rep_locals, locals, local_val, match_local], grad_list = self.findSEPs()
        nOfLocals = locals.shape[0]
        self.locals = locals  #####transpose
        self.match_local = match_local
        # % Find transition points and label the SEPs
        self.findTPs()
        self.grad_list=grad_list

        # %% Cluster assignment of each data point

        # --- Automatic determination of cluster number based on the cluster boundary
        if not (fHierarchical):

            print('Automatic determination of cluster numbers based on the SVDD boundearies defined by R^2');
            adjacent = np.zeros([nOfLocals, nOfLocals])

            tmp = np.nonzero(self.ts['f'] < self.supportmodel.R)[0]  ########

            if np.nonzero(len(tmp)):  # % only check the connectivity of TSs inside the sphere
                for j in range(len(tmp)):
                    adjacent[self.ts['neighbor'][tmp[j], 0], self.ts['neighbor'][tmp[j], 1]] = 1
                    adjacent[self.ts['neighbor'][tmp[j], 1], self.ts['neighbor'][tmp[j], 0]] = 1
                    # %% To connect nodes which can be connected via directly connected edges.
                for i in range(nOfLocals):
                    for j in range(i):
                        if (adjacent[i, j] == 1):
                            adjacent[i, :] = np.logical_or(adjacent[i, :], adjacent[j, :])
                    adjacent[i, i] = 1
                self.local_clusters_assignments = np.array(cg.connected_components(adjacent)[1])
                print(self.local_clusters_assignments)
                # % model update
            self.ts['x'] = self.ts['x'][tmp, :]
            self.ts['f'] = self.ts['f'][tmp, :]
            self.ts['purturb'] = self.ts['purturb'][tmp, :]
            self.ts['neighbor'] = self.ts['neighbor'][tmp, :]
            self.ts['cuttingLevel'] = self.supportmodel.R

            # % cluster assignment into entire data points
            self.cluster_label = self.local_clusters_assignments[match_local].T  ###transpose
            print(self.cluster_label)
        else:
            self.hierarchicalLabelTSVC()
            print(self.cluster_label)
    def fmsc_induction(self,test_input,normalize = True):
        """
        induct the cluster labels of test input! after fmsc model is trained
        test_input is the number of data x the dimensions
        """
        assert self.labelingmethod == 'F-MSC'
        if normalize:
            n_input = svdd_normalize(test_input.T,self.supportmodel)
            n_input = n_input.T
        else:
            n_input = test_input
        ## Find nearest center to test input
        nbrs = NNs(n_neighbors=1,algorithm='ball_tree').fit(self.centers)
        _,clst = nbrs.kneighbors(n_input)
        cluster_label = self.matchBallIndex(clst,self.fmscmodel['ball_cluster_labels'],self.centers,self.locals).flatten()
        return n_input, cluster_label

    def fmsc(self):
        self.fmscmodel = {}
        self.fmscmodel['options'] = self.options
        self.fmscmodel['support_model'] = self.supportmodel

        #Partitioning Data into Small Ball
        self.R1 = self.options['R1']
        self.R2 = self.options['R2']
        self.partition(self.supportmodel.normalized_input.T, self.R1)

        [all_locals, unique_locals, match_locals] = self.labelFSVC(self.centers, self.supportmodel, self.R2)

        #self.clstmodel['local'] = unique_locals.T
        self.locals = unique_locals
        self.findAdjMatrix(unique_locals.T)
        

        csm = cg.connected_components(self.adjacent_matrix)
        local_clusters_assignments = csm[1]
        local_clusters_assignments = np.array(local_clusters_assignments)



        cluster_labels = local_clusters_assignments[match_locals]
        self.fmscmodel['ball_cluster_labels'] =cluster_labels
        self.local_ass = local_clusters_assignments

        self.cluster_label = self.matchBallIndex(self.clst, cluster_labels, self.centers, unique_locals).flatten()
        #self.clstmodel['centers'] = self.centers
        self.match_local = match_locals

        #print(self.clstmodel['cluster_labels'])


    def partition(self, X, r):
        [N, d] = X.shape
        clst = np.zeros((N,1))
        x_ind = np.arange(0,N,1)
        X2 = X
        i=0
        C = np.empty(shape = [0, d])


        while(True):
            randind = np.random.permutation(X2.shape[0])
            C = np.append(C, np.array([X2[randind[0]]]), axis=0)
            dst = np.sqrt(self.dist2(C[i], X2))
            ind = np.where(dst<r)[1]

            assert ind.size > 0, 'ERROR: R1 is too small! R1 must be positive real number'
            clst[x_ind[ind]] = i

            X2 = np.delete(X2, ind, 0)
            x_ind = np.delete(x_ind, ind, 0)

            i = i+1
            if X2.size == 0:
                break

        self.centers = C
        self.clst = clst

    def labelFSVC(self, centers, model, R2):
        [N, dim] = centers.shape

        N_locals = []
        local_val = []
        arrival = np.zeros([N, 1])

        all_locals = np.array(centers,copy=True)
        converge = 0
        iter = 1

        while(converge == 0):
            ind = np.where(arrival == 0)[0]
            run_Samples = all_locals[ind]

            if model.support == 'SVDD':

                for i in range(len(ind)):
                    x0 = run_Samples[i]
                    if len(x0) <= 2:
                        res = minimize(fun=my_R1, x0=x0, args=model, method='Nelder-Mead')
                        [temp, val] = [res.x, res.fun]
                    else:
                        res = minimize(fun=my_R1, x0=x0, args=model, method='trust-ncg',jac = my_R_grad,hess = my_R_hess)
                        [temp, val] = [res.x, res.fun]
                    all_locals[ind[i]] = temp
                    #local_val[ind[i]] = val

                    if np.sum((temp-x0)*(temp-x0)) < pow(10, -3):
                        arrival[ind[i]][0] = 1

            elif model.support == 'GP':
                for i in range(len(ind)):
                    x0 = run_Samples[i]
                    if len(x0) <= 2:
                        res = minimize(fun=my_R_GP1, x0=x0, args=model, method='Nelder-Mead')
                        [temp, val] = [res.x, res.fun]
                    else:
                        res = minimize(fun=my_R_GP1, x0=x0, args=model, method='trust-ncg')
                        [temp, val] = [res.x, res.fun]
                    all_locals[ind[i]] = temp
                    #local_val[ind[i]] = val

                    if np.sum((temp-x0)*(temp-x0)) < pow(10, -3):
                        arrival[ind[i]][0] = 1

            elif model.support == 'pseudo':

                for i in range(len(ind)):
                    x0 = run_Samples[i]
                    if len(x0) <= 2:
                        res = minimize(fun=my_R_pseudo, x0=x0, args=model, method='Nelder-Mead')
                        [temp, val] = [res.x, res.fun]
                    else:
                        res = minimize(fun=my_R_pseudo, x0=x0, args=model, method='trust-ncg',jac = my_R_pseudo_grad,hess = my_R_pseudo_Hess)
                        [temp, val] = [res.x, res.fun]
                    all_locals[ind[i]] = temp
                    #local_val[ind[i]] = val

                    if np.sum((temp-x0)*(temp-x0)) < pow(10, -3):
                        arrival[ind[i]][0] = 1
                        
            if np.sum(arrival) > 0:
                [all_locals, arrival] =self.mergeBall(all_locals, arrival, R2)

            if np.sum(arrival) == N:
                converge = 1

            iter = iter + 1

        unique_locals, I, match_locals = np.unique(np.round(10 * all_locals), axis=0, return_index=True, return_inverse=True)
        unique_locals = all_locals[I]

        #local_val = local_val[I]

        return [all_locals, unique_locals, match_locals]

    def mergeBall(self, all_locals, arrival, R2):
        N = all_locals.shape[0]
        Done = np.zeros([N, 1])
        State = all_locals

        #R2 = self.options.R2

        for i in range(N):
            if Done[i, 0] == 0:
                Done[i, 0] = 1
                cur_sample = all_locals[i]

                inds = np.where(Done == 0)[0]
                dst = np.sqrt(self.dist2(cur_sample, all_locals[inds]))  ######dist2
                inds2 = np.where(dst < R2)[1]

                if len(inds2) > 0 and arrival[i][0] == 1:

                    inds3 = np.where(arrival[inds[inds2]] == 0)[0]
                    if len(inds3) > 0:
                        State[[inds[inds2[inds3]]]] = np.tile(cur_sample, [len(inds3), 1])
                        arrival[[inds[inds2[inds3]]]] = np.ones([len(inds3), 1])
                        Done[[inds[inds2[inds3]]]] = np.ones([len(inds3), 1])

                elif len(inds2) > 0 and arrival[i][0] == 0:
                    inds4 = np.where(arrival[inds[inds2]] == 1)[0]
                    if len(inds4) == 0:
                        State[inds[inds2]] = np.tile(cur_sample, [len(inds2), 1])
                        Done[inds[inds2]] = np.ones([len(inds2), 1])
                    else:
                        State[i] = all_locals[inds[inds2[inds4[0]]]]
                        arrival[i] = 1

        return [State, arrival]

    def dist2(self, x, c):

        if len(x.shape) == 1:
            ndata = 1
            dimx = x.shape[0]
        else :
            [ndata, dimx] = x.shape

        if len(c.shape) == 1:
            ncentres = 1
            dimc = c.shape[0]
        else:
            [ncentres, dimc] = c.shape
        assert dimx == dimc, "ERROR : Data dimension does not match dimension of centres"

        n2 = (np.ones([ncentres, 1]).dot(np.sum((x*x).T, axis=0).reshape(1, ndata))).T + np.ones([ndata, 1]).dot(np.sum((c*c).T, axis=0).reshape(1, ncentres)) - 2 * (x.dot(c.T))

        if len(np.where(n2<0)[0]) != 0:
            minus_ind = np.where(n2<0)
            for ind in range(len(minus_ind[0])):
                n2[minus_ind[0][ind]][minus_ind[1][ind]] = 0

        return n2

    def matchBallIndex(self, clst, label, C, EVs):
        [n1, d1] = C.shape
        [n2, d2] = EVs.shape
        final_label = np.zeros([clst.shape[0], 1])

        for i in range(0, n1):
            ind = np.where(clst == i)[0]
            final_label[ind] = label[i]


        return final_label


def my_R1(x, model):
    f = kradius(x.reshape(x.shape[0], 1), model)
    return f

def my_R1_cond(x, models):
    f = []
    n_total = 0
    for i in range(len(models)):
        n_total+=models[i].n
    for i in range(len(models)):
        model = models[i]
        f.append((kradius(x.reshape(x.shape[0], 1), model)-model.R)*models[i].n/n_total)
    f = np.min(f)
    return f

def my_R_grad(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']
    f = kradius(x.reshape(x.shape[0], 1), model)

    q = 1 / (2 * model.svdd_model['options']['arg'] * model.svdd_model['options']['arg'])
    K = kernel(model.svdd_model['sv']['X'], model.svdd_model['options']['ker'], model.svdd_model['options']['arg'],
               input2=x.reshape(x.shape[0], 1))
    g = 4 * q * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T,
                       np.multiply(ml.repmat(K, 1, d),
                                   (ml.repmat(x, n, 1) - model.svdd_model['sv']['X'].T)))

    return g

def my_R_hess(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']
    f = kradius(x.reshape(x.shape[0], 1), model)

    q = 1 / (2 * model.svdd_model['options']['arg'] * model.svdd_model['options']['arg'])
    K = kernel(model.svdd_model['sv']['X'], model.svdd_model['options']['ker'], model.svdd_model['options']['arg'],
               input2=x.reshape(x.shape[0], 1))
    g = 4 * q * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T,
                       np.multiply(ml.repmat(K, 1, d),
                                   (ml.repmat(x, n, 1) - model.svdd_model['sv']['X'].T)))

    const = np.multiply(model.svdd_model['Alpha'], K)
    H = []

    for i in range(d):
        H.append(- 8 * q * q * np.sum(np.multiply(np.multiply(ml.repmat(const.T, d, 1), (
        ml.repmat(x[i], d, n) - ml.repmat(model.svdd_model['sv']['X'][i, :].T, d, 1))), (
                                                  ml.repmat(x.reshape(x.shape[0], 1), 1, n) -
                                                  model.svdd_model['sv']['X'])), axis=1).T)
    H = np.array(H).T
    H = H + 4 * q * np.eye(d) * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T, K)
    return H


def my_R2(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']
    f = kradius(x.reshape(x.shape[0], 1), model)

    q = 1 / (2 * model.svdd_model['options']['arg'] * model.svdd_model['options']['arg'])
    K = kernel(model.svdd_model['sv']['X'], model.svdd_model['options']['ker'], model.svdd_model['options']['arg'],
               input2=x.reshape(x.shape[0], 1))
    g = 4 * q * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T,
                       np.multiply(ml.repmat(K, 1, d),
                                   (ml.repmat(x, n, 1) - model.svdd_model['sv']['X'].T)))

    const = np.multiply(model.svdd_model['Alpha'], K)
    H = []

    for i in range(d):
        H.append(- 8 * q * q * np.sum(np.multiply(np.multiply(ml.repmat(const.T, d, 1), (
        ml.repmat(x[i], d, n) - ml.repmat(model.svdd_model['sv']['X'][i, :].T, d, 1))), (
                                                  ml.repmat(x.reshape(x.shape[0], 1), 1, n) -
                                                  model.svdd_model['sv']['X'])), axis=1).T)
    H = np.array(H).T
    H = H + 4 * q * np.eye(d) * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T, K)
    return [f, g, H]

    

def my_R_GP1(x, supportmodel):
    model = supportmodel
    f = var_gpr(np.reshape(x, [-1, 1]), model.normalized_input, model.inv_C, model.gp_params)  ##full(X)
    #    %[n d]=size(model.SVs);
    #    %q=model.Parameters(4);
    #    %SV=full(model.SVs);
    #    %beta=model.sv_coef;
    return f

def my_R_GP2(x, model):
    # %
    # %   Calculating function value and gradient of
    # %   the trained kernel radius function
    # %==========================================================================
    # % Implemented by Kyu-Hwan Jung at April 26, 2010.
    # % Modified by Sujee Lee at September 3, 2014.
    # %
    # % * The source code is available under the GNU LESSER GENERAL PUBLIC
    # % LICENSE, version 2.1.
    # %==========================================================================
    x = x.reshape(-1, 1)

    input = model.normalized_input
    inv_K = model.inv_C
    hparam = model.gp_params

    f = var_gpr(x, input, inv_K, hparam)  ##############full

    #    %[n d]=size(model.SVs);
    #    %q=model.Parameters(4);
    #    %SV=full(model.SVs);
    #    %beta=model.sv_coef;
    input = input.T
    x = x.reshape(1, -1)
    hparam = np.append(hparam[0], np.array(hparam[1:]).reshape(-1, 1), axis=0)
    [N, D] = input.shape
    [nn, D] = x.shape  # % nn=1

    k = np.zeros((N, nn))
    for d in range(D):
        k += hparam[d][0] * np.power((ml.repmat(input[:, d].reshape(-1, 1), 1, nn) - ml.repmat(np.transpose(x[:, d]), N, 1)), 2)
    k = hparam[D] * np.exp(-0.5 * k)
    gk = -np.multiply(np.multiply(ml.repmat(k, 1, D), ml.repmat(np.transpose(hparam[0:D]), N, 1)), ml.repmat(x, N, 1) - input)
    g = -2 * np.dot(np.dot(np.transpose(gk), inv_K), k)

    Hk = np.zeros((N, D ** 2))  # % Hessian of k. (make N * D^2 matrix)
    for j in range(D):
        Hkj = np.multiply(gk , -hparam[j] * ml.repmat((x[:, j] - input[:, j]), D, 1).T ) # % x(:,j)-input(:,j) : scalar - vector
        Hkj[:, j] = Hkj[:, j] - hparam[j] * k.reshape(1, -1)
        Hk[:, j * D:(j + 1) * D] = Hkj  #####################index
    H1 = np.matmul(np.matmul(Hk.T, inv_K), k)  # % D^2 * 1
    H1 = np.reshape(H1, [D, D])  # % D * D
    H2 = np.matmul(np.matmul(gk.T, inv_K), gk)  # % D * D
    H = -2 * (H1 + H2)
    return f, g, H


def my_R_pseudo(x, model):
    f_=0
    pmodel=model.pseudo_model
    arg=model.pseudo_params['arg']
    gamma=0.5 / (arg **2)

    f_ = np.dot(pmodel['Alpha'].T, np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))
    f = -np.log(f_)
    return f
def my_R_pseudo_grad(x,model):
    f_=0
    d=x.shape[0]
    pmodel=model.pseudo_model
    arg=model.pseudo_params['arg']
    gamma=0.5 / (arg **2)
    f_ = np.dot(pmodel['Alpha'].T, np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))
    f = -np.log(f_)
    g = np.zeros(d)
    g = -2*gamma*np.dot(pmodel['Alpha'].T, (np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)*np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))/f_
    g = g.flatten()
    return g
def my_R_pseudo_Hess(x,model):
    d=x.shape[0]
    f_ = 0
    pmodel=model.pseudo_model
    arg=model.pseudo_params['arg']
    gamma=0.5 / (arg **2)
    f_ = np.dot(pmodel['Alpha'].T, np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))
    if np.abs(f_)<10**-300:
        f_ = np.array([[10**-300]])
    f = -np.log(f_)
    g = np.zeros(d)
    g = -2*gamma*np.dot(pmodel['Alpha'].T, (np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)*np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))/f_
    H = np.zeros([d,d])
    v = np.dot((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T).T, np.diag(np.ravel(pmodel['Alpha'].T*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)))))
    w = np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T
    H = 4*gamma**2/f_* np.dot(v,  w)
    H = H + np.diag(-2*gamma*np.ones([d,]))    
    return H

def my_R2_pseudo(x, model):
    ##### lsy
    d=x.shape[0]
    #print(x.shape) #[2,]
    
    f_ = 0
    pmodel=model.pseudo_model
    arg=model.pseudo_params['arg']
    gamma=0.5 / (arg **2)
    #print(np.tile(x[0],[pmodel['nsv'],1]))
    #print(np.tile(x[0],[pmodel['nsv'],1]).shape)
    #print(pmodel['sv']['X'].T.shape)
    f_ = np.dot(pmodel['Alpha'].T, np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))
    if np.abs(f_)<10**-300:
        f_ = np.array([[10**-300]])
    print(x)
    print(f_)
    #print(pmodel['sv']['X'])
    f = -np.log(f_)
    
    g = np.zeros(d)
    #print('.')
    #print(np.expand_dims(np.tile(x[0],[pmodel['nsv'],1])-pmodel['sv']['X'].T, 1).shape)
    #print(np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1).shape) #[35,1]
    #print(pmodel['Alpha'].T.shape) #[1,35]
    #for i in range(d):
    #    g[i] = -2*gamma*np.dot(pmodel['Alpha'].T, (np.tile(x[i],[pmodel['nsv'],1])-pmodel['sv']['X'].T)*np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))/f_
    g = -2*gamma*np.dot(pmodel['Alpha'].T, (np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)*np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))/f_
    #print(g.shape)
    H = np.zeros([d,d])
    #for i in range(d):
    #    for j in range(d):
    #        H[i][j]=4*gamma**2*np.dot(pmodel['Alpha'].T, (np.tile(x[j],[pmodel['nsv'],1])-pmodel['sv']['X'].T)*(np.tile(x[i],[pmodel['nsv'],1])-pmodel['sv']['X'].T)*np.expand_dims(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)),1))/f_
    #        
    #        if i==j:
    #            H[i][j]=H[i][j]-2*gamma
    #print(pmodel['Alpha']*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)))
    #print((pmodel['Alpha']*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1))).shape)
    #print(pmodel['Alpha'].shape) #[35,1]
    #print(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)).shape) #[35,]
    #print((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T).T.shape) #[2,35]
    #print((pmodel['Alpha']*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1))).shape) #[35,35]
    #print(np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)).shape) #[35,]
    #print(np.ravel(pmodel['Alpha']*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1))).shape)
    #print(np.diag(np.ravel(pmodel['Alpha']*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)))).shape)
    v = np.dot((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T).T, np.diag(np.ravel(pmodel['Alpha'].T*np.exp(-gamma*np.sum(((np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T)**2), 1)))))
    w = np.tile(x,[pmodel['nsv'],1])-pmodel['sv']['X'].T
    #print(v.shape) #[2,35]
    #print(w.shape) #[35,2]

    H = 4*gamma**2/f_* np.dot(v,  w)
    H = H + np.diag(-2*gamma*np.ones([d,]))    
    return f, g, H

def fsolve_R(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']

    q = 1 / (2 * model.svdd_model['options']['arg'] ** 2)
    K = kernel(model.svdd_model['sv']['X'], ker=model.svdd_model['options']['ker'],
               arg=model.svdd_model['options']['arg'], input2=x.reshape(x.shape[0], 1))

    F = 4 * q * np.dot(np.transpose(model.svdd_model['Alpha']),
                       np.multiply(ml.repmat(K, 1, d), ml.repmat(x, n, 1) - np.transpose(model.svdd_model['sv']['X'])))
    return F[0]

def fsolve_R_GP(x, model):
    input = model.normalized_input.T
    hparam = model.gp_params

    [N, D] = input.shape
    x = x.reshape(-1, D)
    [nn, D] = x.shape

    inv_K = model.inv_C
    hparam = np.append(hparam[0], np.array(hparam[1:]).reshape(-1,1), axis=0)

    k = np.zeros((N, nn))
    for d in range(D):
        k += hparam[d][0] * np.power((ml.repmat(input[:, d].reshape(-1, 1), 1, nn)-ml.repmat(np.transpose(x[:, d]), N, 1)), 2)
    k = hparam[D] * np.exp(-0.5 * k)
    gk = -np.multiply(np.multiply(ml.repmat(k, 1, D), ml.repmat(np.transpose(hparam[0:D]), N, 1)),
                      ml.repmat(x, N, 1) - input)
    g = -2 * np.dot(np.dot(np.transpose(gk),inv_K), k)
    return g.ravel()

def fsolve_R_pseudo(x, model):
    ### svdd for the moment
    d = x.shape[0]
    n = model.pseudo_model['nsv']

    q = 1 / (2 * model.pseudo_model['options']['arg'] ** 2)
    K = kernel(model.pseudo_model['sv']['X'], ker=model.pseudo_model['options']['ker'],
               arg=model.pseudo_model['options']['arg'], input2=x.reshape(x.shape[0], 1))

    F = 4 * q * np.dot(np.transpose(model.pseudo_model['Alpha']),
                       np.multiply(ml.repmat(K, 1, d), ml.repmat(x, n, 1) - np.transpose(model.pseudo_model['sv']['X'])))
    return F[0]