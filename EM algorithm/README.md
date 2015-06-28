# Expectation-Maximization algorithm

These files provide a Python implementation of the EM algorithm for a mixture of K Gaussians in a d-dimensional space (K=4 and d=2 for the included datasets), for i.i.d data.

The following algorithms are implemented :

- K-means algorithm, with graphical representation of clusters, and distortion measures
- EM algorithm for a Gaussian mixture with covariance matrices proportional to identity (with K-means initialization) ; a graphical representation of clusters shows ellipses containing a certain mass proportion of the Gaussian distribution
- EM algorithm for a Gaussian mixture with general covariance matrices, with graphical representation of the estimated latent variables
- Log-likelihoods of these mixture models both on training and testing sets

The main function (main.py) implements these algorithms, prints and plots all results.