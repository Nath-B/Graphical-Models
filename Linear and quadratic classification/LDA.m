%% LDA MODEL

% Import training data

train=importdata(path_train);
n=size(train,1);
m=size(train,2);

% Define X_train and Y_train

X_train = train(:,1:m-1);
Y_train = train(:,m);

% Define Xpos and Xnull for labels 1 and 0

Xpos=X_train(find(Y_train==1),:);
Xnull=X_train(find(Y_train==0),:);

% Define maximum likelihood estimators for the training set

p_hat=mean(Y_train);
mu_1_hat=sum(Xpos)/size(Xpos,1);
mu_0_hat=sum(Xnull)/size(Xnull,1);

sigma_1_hat=(Xpos-ones(size(Xpos,1),1)*mu_1_hat)'*(Xpos-ones(size(Xpos,1),1)*mu_1_hat)/size(X_train,1);
sigma_0_hat=(Xnull-ones(size(Xnull,1),1)*mu_0_hat)'*(Xnull-ones(size(Xnull,1),1)*mu_0_hat)/size(X_train,1);
sigma_hat=sigma_1_hat+sigma_0_hat;

% Define the parameters w and b used to compute p(y=1 knowing x)

w=inv(sigma_hat)*(mu_1_hat-mu_0_hat)';
b=(mu_0_hat*inv(sigma_hat)*mu_0_hat'-mu_1_hat*inv(sigma_hat)*mu_1_hat')/2 ...
    +log(p_hat/(1-p_hat));

% Misclassification error for training set

p_lim = 0.5;

predicted_label=1./(1+exp(-w(1)*X_train(:,1)-w(2)*X_train(:,2)-b));
predicted_label_pos=1./(1+exp(-w(1)*Xpos(:,1)-w(2)*Xpos(:,2)-b));
predicted_label_null=1./(1+exp(-w(1)*Xnull(:,1)-w(2)*Xnull(:,2)-b));

mispredicted_pos=sum(predicted_label_pos<p_lim);
mispredicted_null=sum(predicted_label_null>=p_lim);
classification_error=(mispredicted_null+mispredicted_pos)*100/size(X_train,1);

% Define the contour

loc_x= [-8:.01:8];
loc_y=[-8:.01:8];
[grid_x,grid_y] = meshgrid(loc_x,loc_y);
function_values=1./(1+exp(-w(1)*grid_x-w(2)*grid_y-b));

% Plot the points

figure;
scatter(Xpos(:,1),Xpos(:,2),'b','filled'); hold on,
scatter(Xnull(:,1),Xnull(:,2),'r','filled'); hold on,
scatter(Xpos(predicted_label_pos<p_lim,1),Xpos(predicted_label_pos<p_lim,2),'c','filled'); hold on,
scatter(Xnull(predicted_label_null>=p_lim,1),Xnull(predicted_label_null>=p_lim,2),'c','filled'); hold on,
title(['Set ' set ': Decision boundary for a Generative model, training set   [' ...
 num2str(classification_error) ' %]']);
legend('label=1','label=0','misclassified');
xlabel('x');
ylabel('y');

% Plot the contour

[C,h] = contour(grid_x,grid_y,function_values,[p_lim,p_lim]);
clabel(C,h);

% Display the classification error

fprintf(['----- GENERATIVE MODEL (training set ' set ') ----']); fprintf('\n');

% Display the parameters
disp(['w(1)=' num2str(w(1)) '; w(2)=' ...
num2str(w(2)) '; b=' num2str(b)]);
string = 'Misclassification error : ';
string = [string num2str(classification_error)];
string= [string ' %'];
disp(string);
disp([num2str(mispredicted_pos) ' false nulls']);
disp([num2str(mispredicted_null) ' false positives']);
fprintf('\n');