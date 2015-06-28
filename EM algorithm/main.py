# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from Kmeans import Kmeans, Kmeansinit
from EM1 import EMiso
from EM2 import EMgeneral
from likelihood import loglikelihood

# Import all data
train=pd.read_table("EMGaussian.data",header=-1,sep=" ")
test=pd.read_table("EMGaussian.test",header=-1,sep=" ")
X_train=np.array(train)
X_test=np.array(test)

# Run Kmeans once
[centers,clusters,value]=Kmeans(X_train)

## Run Kmeans several times to get a K-means initialization for EM
## Comment the plot/print commands within Kmeans.py for a large number of iterations
nb_it=100
[mu_prec,pi,sigma, value]=Kmeansinit(nb_it,X_train)

## Run EM in the diagonal covariance matrices case, taking a Kmeans initialization
[mu2,pi2,sigma2]=EMiso(mu_prec,pi,X_train)

## Compute likelihood for X_train and X_test in this specific case
marg_train=loglikelihood(X_train,mu2,sigma2,pi2)
print(marg_train)

marg_test=loglikelihood(X_test,mu2,sigma2,pi2)
print(marg_test)

## Run EM2 in the general covariance matrices case, taking a Kmeans initialization
[mu_f,pi_f,sigma_f]=EMgeneral(mu_prec,pi,sigma,X_train)

## Compute the marginal log-likelihood for X_train and X_test
marg_train=loglikelihood(X_train,mu_f,sigma_f,pi_f)
print(marg_train)

marg_test=loglikelihood(X_test,mu_f,sigma_f,pi_f)
print(marg_test)