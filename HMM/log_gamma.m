function g = log_gamma(log_alpha_T,log_beta_T)
% Returns the logarithm of the posterior probabilities gamma
% for every time step

% Initialize number of clusters
K=4

max_alpha_beta=max(log_alpha_T+log_beta_T,[],2);

reduced_ab=log_alpha_T+log_beta_T-repmat(max_alpha_beta,[1,K]);
tmp=log(sum(exp(reduced_ab),2));

g=log_alpha_T+log_beta_T-repmat(max_alpha_beta,[1,K])-repmat(tmp,[1,K]);

end

