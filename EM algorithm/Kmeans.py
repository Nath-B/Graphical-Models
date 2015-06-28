# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def Kmeans(X_trainf,n_clusters=4):
     ''' This function implements the K-means algorithm
     on a multi-labeled dataset
     '''       
    # Copy the training data to keep them still
    X_train=X_trainf.copy()    
    
    # Initialize clusters
    clusters=np.zeros(len(X_train))
    # Draw centroids at random among the data points
    random_points=np.random.randint(0,len(X_train),n_clusters)
    centers_prec=X_train[random_points,:]
    
    # Initialize value V
    V="unknown"
    
    while 1 :
        for k in range(len(X_train)):
            X_k=X_train[k,:]
            distances = np.zeros(n_clusters)
            
            # Define the distances between X_k and the centers
            for i in range(n_clusters):
                diff_i=X_k-centers_prec[i,:]
                distances[i]=np.sqrt(sum(diff_i**2))
            
            # Assign X_k to the closest centroid
            clusters[k]=np.argmin(distances)
        
        # Plot the centroids as well as the clusters
        # Here we only plot 4 clusters
        plt.scatter(X_train[clusters==0,0],X_train[clusters==0,1],c='purple')
        plt.scatter(X_train[clusters==1,0],X_train[clusters==1,1],c='green')
        plt.scatter(X_train[clusters==2,0],X_train[clusters==2,1],c='blue')
        plt.scatter(X_train[clusters==3,0],X_train[clusters==3,1],c='yellow')
        plt.plot(centers_prec[:,0],centers_prec[:,1],'ro')
        plt.axis([-17, 17, -17, 17])
        plt.title("Clusters obtained with Kmeans")
        plt.legend(['Centroids','cluster 1','cluster 2','cluster 3','cluster 4'])
        plt.show()
            
        # Define the new centroids  
        centers=np.zeros((n_clusters,2))
        for i in range(n_clusters):
            if len(X_train[clusters==i])==0 : break 
            centers[i,:]=sum(X_train[clusters==i])/len(X_train[clusters==i])
            
        # CONVERGENCE CRITERION : break the while loop if convergence
        # is achieved, i.e when centers stay still
        diff=centers_prec-centers   
        if diff.all()==0 :  break
        
        # Update centers_prec
        centers_prec=centers
        # Compute the value that we want to minimize
        V=0.    
        for i in range(n_clusters):
            if len(X_train[clusters==i])==0 :
                print('The right number of clusters has not been found')
                return centers, clusters,-1
            diff_i=X_train[clusters==i,:]-centers[i,:]
            V=V+sum(sum(diff_i**2))
        V=V/len(X_train)

        print('Final distortion is : ',V)
    
    # End of the loop : print final distorsion measure
    print('Final distortion is : ',V)
    # Print the centers' coordinates
    print('The coordinates of clusters centers are :','\n',centers_prec)
    return centers,clusters,V

def Kmeansinit(nb_it,X_train,n_clusters=4):
     ''' This function implements a K-means initialization on a training set :
     it returns the optimal clusters parameters obtained for a given number of iterations
     '''          
    ## Run Kmeans several times to get a K-means initialization for EM  
    [centers, clusters, value]=Kmeans(X_train)
    value=1000
    
    # Find the lowest value for different random initializations
    for k in range(nb_it):
        [centers_k,clusters_k,value_k]=Kmeans(X_train)
        if (value_k>0 and value_k<value) :
            centers=centers_k.copy()
            clusters=clusters_k.copy()
            value=value_k.copy()
    
    ## Find the values of the parameters pi, sigma and mu
    mu=centers.copy()
    pi=np.zeros((n_clusters,1))
    sigma=np.zeros((n_clusters,2,2))
    for i in range(n_clusters):
        pi[i]=len(X_train[clusters==i])/len(X_train)
        diff_i=X_train[clusters==i,:]-mu[i,:]
        sigma[i]=np.dot(np.transpose(diff_i),diff_i)/len(X_train[clusters==i])
        
    ## Return the values of mu, pi and sigma
    return mu,pi,sigma,value