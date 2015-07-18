# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

## EM function, takes intial parameters mu, pi and sigma as inputs
def EMgeneral(mu_precf,pif,sigmaf,X_trainf,n_clusters=4):
     ''' This function implements the Expectation-Maximization algorithm
     on a multi-labeled dataset, with general clusters covariance matrices
     '''    
    # Initialize sigma, pi, mu from the values obtained with 
    # Kmeans initialization and given as inputs
    sigma=sigmaf.copy()
    pi=pif.copy()
    mu_prec=mu_precf.copy()
    X_train=X_trainf.copy()
    
    #Initialize the latent variables q
    q=np.zeros((n_clusters,len(X_train),1))
      
    while 1:
        
        '''Expectation step'''
        # First compute a matrix with the terms needed to compute q_ik
        denom=np.zeros((len(X_train),n_clusters))
        for j in range(n_clusters):
            for i in range(len(X_train)):
                    inv_j=np.linalg.inv(sigma[j])
                    detj=np.linalg.det(sigma[j])
                    diff_j=np.matrix(X_train[i,:]-mu_prec[j,:])
                    a_ij=1/(2*np.pi*np.sqrt(detj))*np.exp(-1/2*diff_j*inv_j*np.transpose(diff_j))
                    b_ij=pi[j]*a_ij
                    denom[i,j]=b_ij
                    
        # Then compute the values q_ik 
        for k in range(n_clusters):
            for i in range(len(X_train)):
                q[k][i]=denom[i,k]/sum(denom[i,:])
        
        '''Maximization step'''
        mu=np.zeros((n_clusters,2))
        for k in range(n_clusters):
            mu[k,:]=sum(X_train*q[k])/sum(q[k])
            pi[k]=sum(q[k])/sum(sum(q))
            diff=X_train-mu[k,:]
            sigma[k]=np.dot(np.transpose(diff),diff*q[k])/sum(q[k])
            
        '''Convergence criterion'''
        epsilon=0.0001
        if sum(sum((mu_prec-mu)**2))<epsilon : break
            
        # Update the value of mu_prec
        mu_prec=mu
        print('\n')
        
        
        # Plot the ellipses containing 90% of the distribution
        delta = 0.1
        x = np.arange(-15, 15, delta)
        y = np.arange(-15, 15, delta)
        X, Y = np.meshgrid(x, y)
        Z=np.zeros((n_clusters,300,300))
        
        alpha=90/100 # Confidence parameter
        for k in range(n_clusters):
            inv_k=np.linalg.inv(sigma[k])
            Z[k]=(X-mu[k,0])**2*inv_k[0,0]+(Y-mu[k,1])**2*inv_k[1,1]+2*(X-mu[k,0])*(Y-mu[k,1])*inv_k[0,1]
            # When (X,Y) is a bi-dimensional gaussian,
            # Z has distribution Ki-squared(2)
            p=stats.chi2.ppf(alpha,2)
            plt.contour(X,Y,Z[k],[p],linewidths = 2)
         
        ## Plot the points according to the cluster determined by the
        ## maximum latent variable q
        ## Here we plot 4 clusters
        cluster_1=(q.argmax(axis=0)==0).ravel()
        cluster_2=(q.argmax(axis=0)==1).ravel()
        cluster_3=(q.argmax(axis=0)==2).ravel()
        cluster_4=(q.argmax(axis=0)==3).ravel()
        
        plt.scatter(X_train[cluster_1,0],X_train[cluster_1,1],c='yellow')
        plt.scatter(X_train[cluster_2,0],X_train[cluster_2,1],c='purple')
        plt.scatter(X_train[cluster_3,0],X_train[cluster_3,1],c='green')
        plt.scatter(X_train[cluster_4,0],X_train[cluster_4,1],c='pink')
        plt.plot(mu[:,0],mu[:,1],'ro')         
            
        plt.axis([-10,10,-15,15])
        plt.title("EM algorithm with general covariance matrices")
        plt.show()
        
    ## Return the values of mu, pi and sigma
    return mu_prec, pi, sigma