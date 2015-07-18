function [Pi_clust,q,mu,sigma,Lc] = EM(X,K,runs)
% Run the Expectation-Maximization algorithm given a dataset, a number of clusters,
% and a number of Kmeans runs for the initialization

[m,n] = size(X);

% Initialize latent variables
q=ones(m,K);
% Initialize centers, covariance matrices and cluster probabilities
% with Kmeans
[mu,sigma,~,Pi_clust,~] = Kmeans_runs(X,K,runs);
% Stop the algorithm after n_iterations iterations
n_iterations=100;
% Initialize complete observation log-likelihood
Lc=zeros(1,n_iterations);

% Run the main EM loop
for t=1:n_iterations
    for k=1:K
        % E-step
        q(:,k) = Pi_clust(k)*mvnpdf(X,mu(k,:),sigma(:,:,k));
        norm=0;
        for j=1:K
            norm=norm+Pi_clust(j)*mvnpdf(X,mu(j,:),sigma(:,:,j));
        end
        q(:,k)=q(:,k)./norm;
        Pi_clust(k) = mean(q(:,k));

        % M-step
        mu(k,:) = sum((q(:,k)*ones(1,n)).*X)/sum(q(:,k));
        sigma(:,:,k) = ((q(:,k)*ones(1,n))'.*(X-ones(m,1)*mu(k,:))')*(X-ones(m,1)*mu(k,:))/sum(q(:,k));
    
        % Compute the expected complete observation log-likelihood
        l=0;
        for j=1:K
           l=l+q(:,j).*log(Pi_clust(j)*mvnpdf(X,mu(j,:),sigma(:,:,j)));
        end
        Lc(1,t)=sum(l,1)/m;
    
    end
end

end