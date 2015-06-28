# -*- coding: utf-8 -*-

import numpy as np

def loglikelihood(X_train,mu,sigma,pi,n_clusters=4):
     ''' This function computes the log-likelihood of the gaussian-distributed data
     described with the input parameters 
     '''     
    marg_train=0
    
    for i in range(len(X_train)):
        marg_train_i=0   
        for k in range(n_clusters):
            diff_k=np.matrix(X_train[i,:]-mu[k,:])
            inv_k=np.linalg.inv(np.matrix(sigma[k]))
            det_k=np.linalg.det(sigma[k])
            N_k=1/(2*np.pi*np.sqrt(det_k))*np.exp(-1/2*diff_k*inv_k*np.transpose(diff_k))
            marg_train_i=marg_train_i+pi[k]*N_k
            
        marg_train=marg_train+np.log(+marg_train_i)
    return marg_train