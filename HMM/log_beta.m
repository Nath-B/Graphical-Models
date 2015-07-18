function b = log_beta(t,A,train,mu,sigma)
% Implements the beta-recursion, returns
% a matrix containing the beta(q_t) values, 
% for q_t iterating over the possible states
% and for all t_0 up to t

% Initialize number of clusters
K=4

T=size(train,1);

b=zeros(T-t+1,K);

if t==T
    b(T-t+1,:)=log(ones(1,K));
else
    proba_cond=zeros(1,K);
    for q_t=1:K
      proba_cond(q_t)=mvnpdf(train(t+1,:),mu(q_t,:),sigma(:,:,q_t));
    end
    
    b(2:end,:)=log_beta(t+1,A,train,mu,sigma);
    max_b=max(b(2,:));
    tmp=exp(b(2,:)-max_b).*proba_cond;
    b(1,:)=max_b+log((A*tmp'))';
end

end

