function a = log_alpha(t,A,pi_0,train,mu,sigma)
% Implements the alpha-recursion, returns
% a matrix containing the alpha_(t_0)(q_t) values, 
% for q_t iterating over the possible states
% and for all t_0 up to t

% Initialize number of clusters
K=4
% Initialize the alpha vector
a=zeros(t,K);

proba_cond=zeros(1,K);
for q_t=1:K
  proba_cond(q_t)=mvnpdf(train(t,:),mu(q_t,:),sigma(:,:,q_t));
end

% Recursively compute alpha
if t==1
    a(t,:)=log(pi_0.*proba_cond);
else
    a(1:(t-1),:)=log_alpha(t-1,A,pi_0,train,mu,sigma);
    max_q_t=max(a(t-1,:));
    a(t,:)=log(proba_cond)+max_q_t+log(exp(a(t-1,:)-max_q_t)*A);
end

end

