function q = quadratic_qda(A,X1,X2)
% Computes the quadratic term for a QDA model

q=A(1,1)*X1.^2+A(2,2)*X2.^2 ...
    +2*A(1,2)*X1.*X2;

end

