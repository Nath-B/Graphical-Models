function [mu,sigma,clusters,Pi_clust,J] = Kmeans_runs(X,K,runs)
% Runs Kmeans several times and outputs the parameters
% associated with the lowest distortion

% Run Kmeans once to initialize parameters
[mu,sigma,clusters,Pi_clust,J] = Kmeans(X,K);

% Run Kmeans several times
for t=1:runs
    [mu_temp,sigma_temp,clusters_temp,P_temp,J_temp] = Kmeans(X,K);
    
    % Keep the values if distortion is lower
    if J_temp(end)<J(end)
       mu = mu_temp;
       sigma = sigma_temp;
       clusters = clusters_temp;
       Pi_clust = P_temp;
       J = J_temp;
    end
end