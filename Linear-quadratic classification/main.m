%% Main script

% Clear all variables and the console

clear all;
clc;

% Define the set we want to use (A,B, or C)

set='B';

% Define the paths from which data has to be taken

path_train=['classification_data/classification' num2str(set) '.train'];
path_test=['classification_data/classification' num2str(set) '.test'];


% Run the different models on the training sets to check training errors

fprintf('-------------------- TRAINING SET --------------------');
fprintf(['\n' '\n']);

LDA;
Logistic;
Linear;
QDA;

% Compute the misclassification errors on the testing sets

fprintf('-------------------- TESTING SET --------------------');
fprintf(['\n' '\n']);

LDA_test_set;
Logistic_test_set;
Linear_test_set;
QDA_test_set;