function [pi_0,A,mu,sigma,loglikelihood] = HMM_EM(nb_iterations,X,K,mu_init,sigma_init,A_init,pi_0_init)
% Implements the EM algorithm for Hidden Markov Models
% Returns pi_0,A,mu,sigma parameters as well as the log-likelihood

% Initialize parameters
A_prev=A_init;
mu_prev=mu_init;
sigma_prev=sigma_init;
pi_0_prev=pi_0_init;
train=X;
loglikelihood=zeros(1,nb_iterations);
[m,n] = size(X);

for t=1:nb_iterations

%% E-step
% Compute the posterior probabilities (gamma values)
log_alpha_T = log_alpha(m,A_prev,pi_0_prev,train,mu_prev,sigma_prev);
log_beta_T=log_beta(1,A_prev,train,mu_prev,sigma_prev);
log_gamma_T=log_gamma(log_alpha_T,log_beta_T);
gamma_T=exp(log_gamma_T);

% Compute the posterior coocurrence probabilities
log_eps_q=log_eps(A_prev,train,mu_prev,sigma_prev,log_alpha_T,log_gamma_T);
eps_q=exp(log_eps_q);

%% Compute the log-likelihood
max_alpha_beta=max(log_alpha_T+log_beta_T,[],2);
reduced_ab=log_alpha_T+log_beta_T-repmat(max_alpha_beta,[1,4]);
tmp=log(sum(exp(reduced_ab),2))+max_alpha_beta;

loglikelihood(t)=tmp(1);

%% M-step
% Update parameters
pi_0_prev=gamma_T(1,:);
A_prev=reshape(sum(eps_q,1),[4,4])./repmat(sum(gamma_T(1:end-1,:),1),[4,1]);

for k=1:K
   mu_prev(k,:)=sum(train.*repmat(gamma_T(:,k),[1,2]))/sum(gamma_T(:,k)); 
   cent_train=train-repmat(mu_prev(k,:),[500,1]);
   sigma_prev(:,:,k)=cent_train'*(cent_train.*repmat(gamma_T(:,k),[1,2]))/sum(gamma_T(:,k));
end

end

pi_0=pi_0_prev;
A=A_prev;
mu=mu_prev;
sigma=sigma_prev;

end

