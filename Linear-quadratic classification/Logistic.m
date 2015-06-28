%% LOGISTIC REGRESSION

% Import training data

train_initial=importdata(path_train);
n=size(train_initial,1);
m=size(train_initial,2);

% Add a dimension to X_train so as to compute the offset for logistic
% regression

train=train_initial;
train(:,m+1)=ones(n,1);
X_train=train(:,[1:m-1,m+1]);
Y_train=train(:,m);

% Define Xpos and Xnull for labels 1 and 0

Xpos=X_train(find(Y_train==1),:);
Xnull=X_train(find(Y_train==0),:);

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

% Misclassification error for training set

p_lim = 0.5;

predicted_label=1./(1+exp(-X_train*w));
predicted_label_pos=1./(1+exp(-Xpos*w));
predicted_label_null=1./(1+exp(-Xnull*w));

mispredicted_pos=sum(predicted_label_pos<p_lim);
mispredicted_null=sum(predicted_label_null>=p_lim);
classification_error=(mispredicted_null+mispredicted_pos)*100/size(X_train,1);

% Define the contour

loc_x= [-8:.01:8];
loc_y=[-8:.01:8];
[grid_x,grid_y] = meshgrid(loc_x,loc_y);
function_values=1./(1+exp(-w(1)*grid_x-w(2)*grid_y-w(3)));

% Plot the points

figure;
scatter(Xpos(:,1),Xpos(:,2),'b','filled'); hold on,
scatter(Xnull(:,1),Xnull(:,2),'r','filled'); hold on,
scatter(Xpos(predicted_label_pos<p_lim,1),Xpos(predicted_label_pos<p_lim,2),'c','filled'); hold on,
scatter(Xnull(predicted_label_null>=p_lim,1),Xnull(predicted_label_null>=p_lim,2),'c','filled'); hold on,
title(['Set ' set ': Decision boundary for a logistic regression, training set   [' ...
 num2str(classification_error) ' %]']);
legend('label=1','label=0','misclassified');
xlabel('x');
ylabel('y');

% Plot the contour

[C,h] = contour(grid_x,grid_y,function_values,[p_lim,p_lim]);
clabel(C,h);

% Display the misclassification error

fprintf(['----- LOGISTIC REGRESSION (training set ' set ') ----']); fprintf('\n');

% Display the parameters
disp(['w(1)=' num2str(w(1)) '; w(2)=' ...
num2str(w(2)) '; b=' num2str(w(3))]);

string = 'Misclassification error : ';
string = [string num2str(classification_error)];
string= [string ' %'];
disp(string);
disp([num2str(mispredicted_pos) ' false nulls']);
disp([num2str(mispredicted_null) ' false positives']);
fprintf('\n');