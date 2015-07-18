function e = log_eps(A,train,mu,sigma,log_alpha_T,log_gamma_T)
% Gives the logarithm of the posterior coocurrence probability

% Initialize number of clusters
K=4

T=size(log_alpha_T,1);
e=zeros(size(log_alpha_T,1)-1,K,K);

for t=1:T-1
    for q_t=1:K
        for q_tt=1:K
    
    e(t,q_t,q_tt)=log(A(q_t,q_tt))+log_gamma_T(t+1,q_tt)...
    +log(mvnpdf(train(t+1,:),mu(q_tt,:),sigma(:,:,q_tt)))...
    +log_alpha_T(t,q_t)-log_alpha_T(t+1,q_tt);

        end
    end
end

end

