 %% Initialization
clc; clear; close all;
warning off
addpath('../feature_extraction/')

%% Read feature files
train_table = readtable('training.csv');
test_table_in = readtable('test_in.csv');
test_table_out = readtable('test_out.csv');

Y_train = log(train_table.Lifetime);
Y_test_in = log(test_table_in.Lifetime);
Y_test_out = log(test_table_out.Lifetime);
%% Dummy model
[~, train_error_dummy,  true_pred_train] = fit_dummy(Y_train, Y_train, true);
[~, test_error_dummy_in,  true_pred_test_in] = fit_dummy(Y_train, Y_test_in, true);
[~, test_error_dummy_out,  true_pred_test_out] = fit_dummy(Y_train, Y_test_out, true);

% Plot results
disp('Dummy Model')
plot_pred_results_fixed_in_out(train_error_dummy,true_pred_train,...
                            test_error_dummy_in,true_pred_test_in,...
                            test_error_dummy_out,true_pred_test_out,...
                            'Dummy Model')

%% Condition model
X_train_cond = table2array(train_table(:,["ChgC_rate","DchgC_rate",'DoD']));
X_test_in_cond = table2array(test_table_in(:,["ChgC_rate","DchgC_rate",'DoD']));
X_test_out_cond =table2array(test_table_out(:,["ChgC_rate","DchgC_rate",'DoD']));
[train_error_cond, true_pred_train_cond, test_error_cond, true_pred_test_in_cond,...
    true_pred_test_out_cond, B_cond, FitInfo_cond] ...
    = fit_elastic_fixed_train_2_test(X_train_cond, Y_train, X_test_in_cond, Y_test_in,...
    X_test_out_cond, Y_test_out, 50, 5, true, false, true, false);

test_error_in_cond = test_error_cond(1,:);
test_error_out_cond = test_error_cond(2,:);
% Plot results
disp('Condition Model')
plot_pred_results_fixed_in_out(train_error_cond,true_pred_train_cond,...
                            test_error_in_cond,true_pred_test_in_cond,...
                            test_error_out_cond,true_pred_test_out_cond,...
                            'Condition Model')

%% Degradation-informed model (2 features)
X_train_di2 = [log(abs(table2array(train_table(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];
X_test_in_di2 = [log(abs(table2array(test_table_in(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];
X_test_out_di2 = [log(abs(table2array(test_table_out(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];

[train_error_di2, true_pred_train_di2, test_error_di2, true_pred_test_in_di2,...
    true_pred_test_out_di2, B_di2, FitInfo_di2] = fit_elastic_fixed_train_2_test(...
    X_train_di2, Y_train, X_test_in_di2, Y_test_in,X_test_out_di2, Y_test_out, ...
    50, 5, true, false, true, false);

test_error_in_di2 = test_error_di2(1,:);
test_error_out_di2 = test_error_di2(2,:);
% Plot results
disp('Degradation-Infomed Model (2 features)')
plot_pred_results_fixed_in_out(train_error_di2,true_pred_train_di2,...
                            test_error_in_di2,true_pred_test_in_di2,...
                            test_error_out_di2,true_pred_test_out_di2,...
                            'Degradation-Infomed Model (2)')

%% Degradation-informed model (3 features)
X_train_di3 = [train_table.DoD,log(abs(table2array(train_table(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];
X_test_in_di3 = [test_table_in.DoD,log(abs(table2array(test_table_in(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];
X_test_out_di3 = [test_table_out.DoD,log(abs(table2array(test_table_out(:,["mean_dqdv_dchg_mid_3_0","delta_CV_time_3_0"]))))];

[train_error_di3, true_pred_train_di3, test_error_di3, true_pred_test_in_di3,...
    true_pred_test_out_di3, B_di3, FitInfo_di3] = fit_elastic_fixed_train_2_test(...
    X_train_di3, Y_train, X_test_in_di3, Y_test_in,X_test_out_di3, Y_test_out, ...
    50, 5, true, false, true, false);

test_error_in_di3 = test_error_di3(1,:);
test_error_out_di3 = test_error_di3(2,:);
% Plot results
disp('Degradation-Infomed Model (3 features)')
plot_pred_results_fixed_in_out(train_error_di3,true_pred_train_di3,...
                            test_error_in_di3,true_pred_test_in_di3,...
                            test_error_out_di3,true_pred_test_out_di3,...
                            'Degradation-Infomed Model (3)')
