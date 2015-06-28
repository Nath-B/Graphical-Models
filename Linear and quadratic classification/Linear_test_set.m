%% LINEAR REGRESSION

% Import training and testing data

train_initial=importdata(path_train);
test_initial=importdata(path_test);
n=size(train_initial,1);
m=size(train_initial,2);
n2=size(test_initial,1);
m2=size(test_initial,2);

% Add a dimension to X_train so as to compute the offset for linear
% regression

train=train_initial;
test=test_initial;
train(:,m+1)=ones(n,1);
X_train=train(:,[1:m-1,m+1]);
Y_train=train(:,m);
test(:,m2+1)=ones(n2,1);
X_test=test(:,[1:m2-1,m2+1]);
Y_test=test(:,m2);

% Define Xpos and Xnull for labels 1 and 0, for training and testing sets

Xpos=X_train(find(Y_train==1),:);
Xnull=X_train(find(Y_train==0),:);
Xpos_test=X_test(find(Y_test==1),:);
Xnull_test=X_test(find(Y_test==0),:);

% Define the parameter w thanks to the training set

w=inv(X_train'*X_train)*X_train'*Y_train;


% Misclassification error for testing set

p_lim = 0.5;

predicted_label=X_test*w;
predicted_label_pos=Xpos_test*w;
predicted_label_null=Xnull_test*w;

mispredicted_pos=sum(predicted_label_pos<p_lim);
mispredicted_null=sum(predicted_label_null>=p_lim);
classification_error=(mispredicted_null+mispredicted_pos)*100/size(X_test,1);

% Define the contour

loc_x= [-8:.01:8];
loc_y=[-8:.01:8];
[grid_x,grid_y] = meshgrid(loc_x,loc_y);
function_values=w(1)*grid_x+w(2)*grid_y+w(3);

% Plot the points

figure;
scatter(Xpos_test(:,1),Xpos_test(:,2),'b','filled'); hold on,
scatter(Xnull_test(:,1),Xnull_test(:,2),'r','filled'); hold on,
scatter(Xpos_test(predicted_label_pos<p_lim,1),Xpos_test(predicted_label_pos<p_lim,2),'c','filled'); hold on,
scatter(Xnull_test(predicted_label_null>=p_lim,1),Xnull_test(predicted_label_null>=p_lim,2),'c','filled'); hold on,
title(['Set ' set ': Decision boundary for a linear regression, testing set   [' ...
 num2str(classification_error) ' %]']);
legend('label=1','label=0','misclassified');
xlabel('x');
ylabel('y');

% Plot the contour

[C,h] = contour(grid_x,grid_y,function_values,[p_lim,p_lim]);
clabel(C,h);

% Display the misclassification error

fprintf(['----- LINEAR REGRESSION (testing set ' set ') ----']); fprintf('\n');
string = 'Misclassification error : ';
string = [string num2str(classification_error)];
string= [string ' %'];
disp(string);
disp([num2str(mispredicted_pos) ' false nulls']);
disp([num2str(mispredicted_null) ' false positives']);
fprintf('\n');