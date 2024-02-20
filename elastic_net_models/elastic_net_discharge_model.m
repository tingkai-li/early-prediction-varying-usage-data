%% Initialization
clc; clear; close all;
warning off
addpath('../feature_extraction/')

%% Read feature files
train_table = readtable('training_discharge.csv');
test_table_in = readtable('test_in_discharge.csv');
test_table_out = readtable('test_out_discharge.csv');

Y_train = log(train_table.Lifetime);
Y_test_in = log(test_table_in.Lifetime);
Y_test_out = log(test_table_out.Lifetime);

X_train = [train_table.Q_ini,log(abs([train_table.min_deltaQ,train_table.var_deltaQ,train_table.kurt_deltaQ,train_table.skew_deltaQ]))];
X_test_in = [test_table_in.Q_ini,log(abs([test_table_in.min_deltaQ,test_table_in.var_deltaQ,test_table_in.kurt_deltaQ,test_table_in.skew_deltaQ]))];
X_test_out = [test_table_out.Q_ini,log(abs([test_table_out.min_deltaQ,test_table_out.var_deltaQ,test_table_out.kurt_deltaQ,test_table_out.skew_deltaQ]))];

disp('Discharge model')
[train_error, true_pred_train, test_error, true_pred_test_in,true_pred_test_out, B, FitInfo] ...
    = fit_elastic_fixed_train_2_test(X_train, Y_train, X_test_in, Y_test_in,X_test_out, Y_test_out, 50, 5, true, false, true, false);

%% Plot results
test_error_in = test_error(1,:);
test_error_out = test_error(2,:);
plot_pred_results_fixed_in_out(train_error,true_pred_train,test_error_in,true_pred_test_in,test_error_out,true_pred_test_out,'Discharge Model')