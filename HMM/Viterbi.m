function Seq = Viterbi(X,K,mu,sigma,A,pi_0)
% Implements the Viterbi algorithm : returns the optimal sequence of states given the input parameters

T=size(X,1);

% Initialize parameters
v=zeros(T,K);

% Initialize v
proba_cond=zeros(1,K);
for q_t=1:K
  proba_cond(q_t)=mvnpdf(X(1,:),mu(q_t,:),sigma(:,:,q_t));
end
v(1,:)=log(pi_0.*proba_cond);

% Compute v recursively   
% and keep track of the states maximizing v at every step
States=zeros(T,K);
for t=2:T
    for j=1:K
    t1=v(t-1,:)+log(A(:,j)');
    t2=log(mvnpdf(X(t,:),mu(j,:),sigma(:,:,j)));
    [v(t,j),States(t,j)]=max(t1+repmat(t2,[1,K]));
    end
end

Seq=zeros(T,1);

% Traceback to find the highest probability path
[~,Seq(T)]=max(v(T,:));
for t=2:T
Seq(T-t+1)=States(T-t+2,Seq(T-t+2));
end

    
end

