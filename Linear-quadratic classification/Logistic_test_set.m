%% LOGISTIC REGRESSION

% Import training and testing data

train_initial=importdata(path_train);
test_initial=importdata(path_test);
n=size(train_initial,1);
m=size(train_initial,2);
n2=size(test_initial,1);
m2=size(test_initial,2);

% Add a dimension to X_train so as to compute the offset for logistic
% regression

train=train_initial;
train(:,m+1)=ones(n,1);
X_train=train(:,[1:m-1,m+1]);
Y_train=train(:,m);
test=test_initial;
test(:,m2+1)=ones(n2,1);
X_test=test(:,[1:m2-1,m2+1]);
Y_test=test(:,m2);

% Define Xpos and Xnull for labels 1 and 0, and Y_train

Xpos=X_train(find(Y_train==1),:);
Xnull=X_train(find(Y_train==0),:);
Xpos_test=X_test(find(Y_test==1),:);
Xnull_test=X_test(find(Y_test==0),:);

% Define the parameter w used to compute p(y=1 knowing x)
% thanks to the Newton Raphson's method

% Initialize

w=zeros(m,1);
k=0;
epsilon=0.01;

% Loop using Newton-Raphson's method

while 1
    k=k+1;
    w_prev=w;
    
    % Jacobian matrix
    J=X_train'*(Y_train-logistic_function(X_train*w_prev));
    
    % Hessian matrix
    D=diag(logistic_function(X_train*w_prev).* ...
        (1-logistic_function(X_train*w_prev)));
    H=-X_train'*D*X_train;
    
    % Compute w and the criterion to be maximized
    w=w_prev-inv(H)*J;
    criterion(k)=sum(logistic_function(X_train*w_prev).*Y_train ...
                 +(1-logistic_function(X_train*w_prev)).*(1-Y_train));
    
    % Convergence criterion
    if abs(sum((w-w_prev).^2)/sum(w.^2))<epsilon
        break
    end
end

% Misclassification error for testing set

p_lim = 0.5;

predicted_label=1./(1+exp(-X_test*w));
predicted_label_pos=1./(1+exp(-Xpos_test*w));
predicted_label_null=1./(1+exp(-Xnull_test*w));

mispredicted_pos=sum(predicted_label_pos<p_lim);
mispredicted_null=sum(predicted_label_null>=p_lim);
classification_error=(mispredicted_null+mispredicted_pos)*100/size(X_test,1);

% Define the contour

loc_x= [-8:.01:8];
loc_y=[-8:.01:8];
[grid_x,grid_y] = meshgrid(loc_x,loc_y);
function_values=1./(1+exp(-w(1)*grid_x-w(2)*grid_y-w(3)));

% Plot the points

figure;
scatter(Xpos_test(:,1),Xpos_test(:,2),'b','filled'); hold on,
scatter(Xnull_test(:,1),Xnull_test(:,2),'r','filled'); hold on,
scatter(Xpos_test(predicted_label_pos<p_lim,1),Xpos_test(predicted_label_pos<p_lim,2),'c','filled'); hold on,
scatter(Xnull_test(predicted_label_null>=p_lim,1),Xnull_test(predicted_label_null>=p_lim,2),'c','filled'); hold on,
title(['Set ' set ': Decision boundary for a logistic regression, testing set   [' ...
 num2str(classification_error) ' %]']);
legend('label=1','label=0','misclassified');
xlabel('x');
ylabel('y');

% Plot the contour

[C,h] = contour(grid_x,grid_y,function_values,[p_lim,p_lim]);
clabel(C,h);

% Display the misclassification error

fprintf(['----- LOGISTIC REGRESSION (testing set ' set ') ----']); fprintf('\n');
string = 'Misclassification error : ';
string = [string num2str(classification_error)];
string= [string ' %'];
disp(string);
disp([num2str(mispredicted_pos) ' false nulls']);
disp([num2str(mispredicted_null) ' false positives']);
fprintf('\n');