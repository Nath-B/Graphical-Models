function [mu,sigma,clusters,Pi_clust,J] = Kmeans(X,K)

[m,n] = size(X);
n_iterations=50;
% Initialize centers
index = floor((m-1)*rand(1,K))+1;
% Draw centroids at random among points
mu = X(index,:);


% Run the Kmeans loop for n_iterations
for t=1:n_iterations
    mu_prev = mu;
    temp_clusters = zeros(m,K);
    
    % Assign each point to the closest centroid
    for i=1:m
        [~,cluster] = min(sum((mu-ones(K,1)*X(i,:)).^2,2));
        temp_clusters(i,cluster) = 1;
    end
    
    % Recompute centroids as centers of masses of clusters
    for k=1:K
        mu(k,:) = temp_clusters(:,k)'*X/sum(temp_clusters(:,k));
    end
    
    % Compute the cluster assigned to each point
    [~,clusters] = max(temp_clusters,[],2);
    
    % Compute the distortion function at iteration t
    S = 0;
    for i=1:m
        S = S + sum((X(i,:)-mu(clusters(i),:)).^2);
    end
    J(1,t)=S/m;
    
    % Break the loop when centers stay still or after n_iterations
    % iterations
    if sum(sum((mu_prev-mu).^2))==0
        break
    end
end

% Compute the covariance matrices
sigma=zeros(n,n,K);
for k = 1:K
    sigma(:,:,k) = ((temp_clusters(:,k)*ones(1,n))'.*(X-ones(m,1)*mu(k,:))')*(X-ones(m,1)*mu(k,:))/sum(temp_clusters(:,k));
end

% Compute the probability of the clusters
Pi_clust = sum(temp_clusters)/sum(sum(temp_clusters));

end

