% Import all data
train=importdata('EMGaussian.data');
test=importdata('EMGaussian.test');
T=size(train,1);
d=size(train,2);

% Plot datapoints
figure;
scatter(train(:,1),train(:,2),'b','filled');

% Find the covariance matrices and means of the gaussian clusters
runs=50;
K=4;
X=train;
[~,q,mu,sigma,~] = EM(X,K,runs);

% Plot the clusters assigned to observations by maximizing 
% the probabilities of cluster assignment, in the case of 4 clusters
[~, index_clusters]=max(q,[],2);
figure();
plot(train(index_clusters==1,1),train(index_clusters==1,2),'go'); hold on,
plot(train(index_clusters==2,1),train(index_clusters==2,2),'bo'); hold on,
plot(train(index_clusters==3,1),train(index_clusters==3,2),'ro'); hold on,
plot(train(index_clusters==4,1),train(index_clusters==4,2),'ko'); hold on,
scatter(mu(:,1),mu(:,2),'ko','filled');
title('Clusters obtained on the training set with a gaussian mixture model');

% Define an example matrix A and pi_0
A=1/6*ones(4,4)+(1/2-1/6)*diag(ones(4,1));
pi_0=ones(1,4)/4;

% Compute the alpha and beta vectors for all t
% We are working with log-values
set(0,'RecursionLimit',1000);
log_alpha_T = log_alpha(T,A,pi_0,train,mu,sigma);
log_beta_T=log_beta(1,A,train,mu,sigma);

log_gamma_T=log_gamma(log_alpha_T,log_beta_T);
gamma_T=exp(log_gamma_T);

%% Compute the posterior coocurrence probabilities
log_eps_q=log_eps(A,train,mu,sigma,log_alpha_T,log_gamma_T);
eps_q=exp(log_eps_q);

%% Plot the posterior probabilities for each of the states
figure
subplot(4,1,1);
plot(gamma_T(1:100,1))
title('State 1')

subplot(4,1,2);
plot(gamma_T(1:100,2))
title('State 2')

subplot(4,1,3);
plot(gamma_T(1:100,3))
title('State 3')

subplot(4,1,4);
plot(gamma_T(1:100,4))
title('State 4')

% Plot the maximum posterior probabilities
[~,q]=max(gamma_T,[],2);
figure()
plot(train(q==1,1),train(q==1,2),'bo'); hold on,
plot(train(q==2,1),train(q==2,2),'ro');hold on,
plot(train(q==3,1),train(q==3,2),'ko');hold on,
plot(train(q==4,1),train(q==4,2),'yo');
title('Clusters obtained with hidden markov model, for training set');

%% Implement the EM algorithm for HMM
runs=100;
K=4;
X=train;
[~,~,mu_init,sigma_init,~] = EM(X,K,runs);
A_init=1/6*ones(4,4)+(1/2-1/6)*diag(ones(4,1));
pi_0_init=ones(1,4)/4;
nb_iterations=50;

[pi_0,A,mu,sigma,loglikelihood] = HMM_EM(nb_iterations,X,K,mu_init,sigma_init,A_init,pi_0_init);

%% Plot the loglikelihood as a function of the iterations of the algorithm
% Training set
plot(loglikelihood,'r');
title('Log-likelihood of the training observations as a function of the iterations of the algorithm')
xlabel('Iterations of the algorithm')
ylabel('Log-likelihood')

% Testing set
X=test;
[pi_0_test,A_test,mu_test,sigma_test,loglikelihood_test] = HMM_EM(nb_iterations,X,K,mu_init,sigma_init,A_init,pi_0_init);
plot(loglikelihood_test,'b');
title('Log-likelihood of the testing observations as a function of the iterations of the algorithm')
xlabel('Iterations of the algorithm')
ylabel('Log-likelihood')

% Both sets
plot(loglikelihood,'r'); hold on
plot(loglikelihood_test,'b');
legend('Training set', 'Testing set');
title('Log-likelihood of the training and testing sets')
xlabel('Iterations of the algorithm')
ylabel('Log-likelihood')

% Return the final log-likelihoods
disp('Log-likelihoods');
disp(['Training set : ' num2str(loglikelihood(end))]);
disp(['Testing set : ' num2str(loglikelihood_test(end))]);

%% For testing set, compute and plot posterior marginal probabilities
set(0,'RecursionLimit',1000);
log_alpha_T_test = log_alpha(T,A,pi_0,test,mu,sigma);
log_beta_T_test=log_beta(1,A,test,mu,sigma);

log_gamma_T_test=log_gamma(log_alpha_T_test,log_beta_T_test);
gamma_T_test=exp(log_gamma_T_test);

n_points = 100;

figure
subplot(4,1,1);
plot(gamma_T_test(1:n_points,1))
title('State 1, test set')

subplot(4,1,2);
plot(gamma_T_test(1:n_points,2))
title('State 2, test set')

subplot(4,1,3);
plot(gamma_T_test(1:n_points,3))
title('State 3, test set')

subplot(4,1,4);
plot(gamma_T_test(1:n_points,4))
title('State 4, test set')

% Plot the maximum posterior probabilities
[~,q_test]=max(gamma_T_test,[],2);
figure()
plot(test(q_test==1,1),test(q_test==1,2),'bo'); hold on,
plot(test(q_test==2,1),test(q_test==2,2),'ro');hold on,
plot(test(q_test==3,1),test(q_test==3,2),'ko');hold on,
plot(test(q_test==4,1),test(q_test==4,2),'yo');
title('Clusters obtained with hidden markov model for the test set');

% Plot the most likely states as a function of time
n_points=100;
figure
plot(q_test(1:n_points));
title('Most likely state as a function of time for the first 100 points, for the test set');

% Implement the Viterbi algorithm on training set
X=train;
Seq = Viterbi(X,K,mu,sigma,A,pi_0);
% Plot the most likely sequence of states on train set
figure
plot(train(Seq==1,1),train(Seq==1,2),'bo'); hold on
plot(train(Seq==2,1),train(Seq==2,2),'ro'); hold on
plot(train(Seq==3,1),train(Seq==3,2),'go'); hold on
plot(train(Seq==4,1),train(Seq==4,2),'yo'); hold on
scatter(mu(:,1),mu(:,2),'k','filled');
title('Clusters obtained with the most likely sequence of states on the train set');

% Implement the Viterbi algorithm on testing set, with
% parameters learnt on training set
X=test;
Seq_test = Viterbi(X,K,mu,sigma,A,pi_0);
% Plot the most likely sequence of states on test set
figure
plot(test(Seq_test==1,1),test(Seq_test==1,2),'bo'); hold on
plot(test(Seq_test==2,1),test(Seq_test==2,2),'ro'); hold on
plot(test(Seq_test==3,1),test(Seq_test==3,2),'go'); hold on
plot(test(Seq_test==4,1),test(Seq_test==4,2),'yo'); hold on
scatter(mu(:,1),mu(:,2),'k','filled');
title('Clusters obtained with the most likely sequence of states on the test set');

% Plot the most likely state as a function of time, for the maximum
% sequence of state obtained with the Viterbi algorithm
n_points=100;
figure
plot(Seq_test(1:n_points));
title('Most likely sequence of states for test set as a function of time for the first 100 points');

% Compare both sequences of states
sum(q_test~=Seq_test)
% Plot most likely states as functions of time
n_points=100;
figure
subplot(2,1,1);
plot(q_test(1:n_points));
title('Most likely state at every iteration step');
subplot(2,1,2);
plot(Seq_test(1:n_points));
title('Most likely sequence of states for test set using the Viterbi algorithm');
