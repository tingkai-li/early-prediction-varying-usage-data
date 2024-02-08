function [training_error, true_pred_train,test_error, true_pred_test_1,true_pred_test_2, B, FitInfo] = fit_elastic_fixed_train_2_test(X_train, Y_train, X_test_1, Y_test_1,X_test_2, Y_test_2, n_sim, cv_, standardize_X, min_max_X, log_target, min_MSE_selection)

% Lay out the default values for the method
arguments
    X_train (:,:) {double}
    Y_train (:,1) {double}
    X_test_1 (:,:) {double}
    Y_test_1 (:,1) {double}
    X_test_2 (:,:) {double}
    Y_test_2 (:,1) {double}

    n_sim = 50 % number of repetitions on hyperparameter optimization
    cv_ = 5 % CV on hyperparameter optimization
    standardize_X = false % Default setting for standardize scaler: False
    min_max_X = true % Default setting for minmax scaler: True
    log_target = true % specify if the output is log - used for error metrics
    min_MSE_selection = false % if false, uses +1 std
end


tic
rng default

training_error = zeros(1,6);
test_error = zeros(2,6);

% If minmax scale has been chosen, scale train and test
if min_max_X==true
    colmin_train = min(X_train);
    colmax_train = max(X_train);

    X_train = rescale(X_train,'InputMax',colmax_train,'InputMin',colmin_train);
    X_test_1 = rescale(X_test_1,'InputMax',colmax_train,'InputMin',colmin_train);
end
% Find the best l1_ratio and corresponding lambda values
% ratio of L1 norm (0, 1], the value close to 0 approaches ridge regression, the value 1 represents lasso, other value represents elastic net.
l1_ratios = linspace(1e-5,1,101);
parfor aaa = 1:length(l1_ratios)
    l1_ratio = l1_ratios(aaa)
    [~,FitInfo_elastic_net] = lasso(X_train, Y_train, ...
        'Alpha', l1_ratio, ....
        'Lambda',linspace(1e-4, 5, 501), ...
        'Standardize', standardize_X, ...
        'CV', cv_, ...
        'MCReps', n_sim, ...
        'Options', statset('UseParallel',true));

    % Read the minimum MSE values for this given l1_ratio
    if min_MSE_selection == true
        min_lambda(aaa) = FitInfo_elastic_net.LambdaMinMSE;
        min_MSE(aaa)=FitInfo_elastic_net.MSE(FitInfo_elastic_net.IndexMinMSE);
    elseif min_MSE_selection == false
        min_lambda(aaa) = FitInfo_elastic_net.Lambda1SE;
        min_MSE(aaa)=FitInfo_elastic_net.MSE(FitInfo_elastic_net.Index1SE);
    end
end

[~,l1_r_idx] = min(min_MSE);
l1_ratio_optimal = l1_ratios(l1_r_idx);
lambda_optimal = min_lambda(l1_r_idx);

% Refit the model
[B,FitInfo] = lasso(X_train,Y_train, ...
    "Alpha",l1_ratio_optimal, ...
    'Lambda',lambda_optimal, ...
    'Standardize', standardize_X);

intercept = FitInfo.Intercept;
y_hat_train = intercept + X_train*B;
y_hat_test_1 = intercept + X_test_1*B;
y_hat_test_2 = intercept + X_test_2*B;
% Calculate error (MAE,MAPE,MSE,MDAE,MDAPE,MDSE), MD=median
if log_target == true
    % Calculate training error
    y_pred_train = exp(y_hat_train);
    y_true_train = exp(Y_train);
    true_pred_train = [y_true_train y_pred_train];

    training_error(1,1) = mean(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MAE
    training_error(1,2) = mean(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1)))*100; % MAPE
    training_error(1,3) = sqrt(mean((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMSE
    training_error(1,4) = median(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MDAE
    training_error(1,5) = median(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MDAPE
    training_error(1,6) = sqrt(median((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMDSE

    % Calculate test error
    y_pred_test = exp(y_hat_test_1);
    y_true_test = exp(Y_test_1);
    true_pred_test_1 = [y_true_test y_pred_test];

    test_error(1,1) = mean(abs(true_pred_test_1(:,1)-true_pred_test_1(:,2))); % MAE
    test_error(1,2) = mean(abs((true_pred_test_1(:,1)-true_pred_test_1(:,2))./true_pred_test_1(:,1)))*100; % MAPE
    test_error(1,3) = sqrt(mean((true_pred_test_1(:,1)-true_pred_test_1(:,2)).^2)); % RMSE
    test_error(1,4) = median(abs(true_pred_test_1(:,1)-true_pred_test_1(:,2))); % MDAE
    test_error(1,5) = median(abs((true_pred_test_1(:,1)-true_pred_test_1(:,2))./true_pred_test_1(:,1))); % MDAPE
    test_error(1,6) = sqrt(median((true_pred_test_1(:,1)-true_pred_test_1(:,2)).^2)); % RMDSE

    y_pred_test = exp(y_hat_test_2);
    y_true_test = exp(Y_test_2);
    true_pred_test_2 = [y_true_test y_pred_test];

    test_error(2,1) = mean(abs(true_pred_test_2(:,1)-true_pred_test_2(:,2))); % MAE
    test_error(2,2) = mean(abs((true_pred_test_2(:,1)-true_pred_test_2(:,2))./true_pred_test_2(:,1)))*100; % MAPE
    test_error(2,3) = sqrt(mean((true_pred_test_2(:,1)-true_pred_test_2(:,2)).^2)); % RMSE
    test_error(2,4) = median(abs(true_pred_test_2(:,1)-true_pred_test_2(:,2))); % MDAE
    test_error(2,5) = median(abs((true_pred_test_2(:,1)-true_pred_test_2(:,2))./true_pred_test_2(:,1))); % MDAPE
    test_error(2,6) = sqrt(median((true_pred_test_2(:,1)-true_pred_test_2(:,2)).^2)); % RMDSE

elseif log_target == false
    % Calculate training error
    true_pred_train = [Y_train y_hat_train];

    training_error(1,1) = mean(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MAE
    training_error(1,2) = mean(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1)))*100; % MAPE
    training_error(1,3) = sqrt(mean((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMSE
    training_error(1,4) = median(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MDAE
    training_error(1,5) = median(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MDAPE
    training_error(1,6) = sqrt(median((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMDSE

    % Calculate test error
    true_pred_test_1 = [Y_test_1 y_hat_test_1];

    test_error(1,1) = mean(abs(true_pred_test_1(:,1)-true_pred_test_1(:,2))); % MAE
    test_error(1,2) = mean(abs((true_pred_test_1(:,1)-true_pred_test_1(:,2))./true_pred_test_1(:,1)))*100; % MAPE
    test_error(1,3) = sqrt(mean((true_pred_test_1(:,1)-true_pred_test_1(:,2)).^2)); % RMSE
    test_error(1,4) = median(abs(true_pred_test_1(:,1)-true_pred_test_1(:,2))); % MDAE
    test_error(1,5) = median(abs((true_pred_test_1(:,1)-true_pred_test_1(:,2))./true_pred_test_1(:,1))); % MDAPE
    test_error(1,6) = sqrt(median((true_pred_test_1(:,1)-true_pred_test_1(:,2)).^2)); % RMDSE

    true_pred_test_2 = [Y_test_2 y_hat_test_2];

    test_error(2,1) = mean(abs(true_pred_test_2(:,1)-true_pred_test_2(:,2))); % MAE
    test_error(2,2) = mean(abs((true_pred_test_2(:,1)-true_pred_test_2(:,2))./true_pred_test_2(:,1)))*100; % MAPE
    test_error(2,3) = sqrt(mean((true_pred_test_2(:,1)-true_pred_test_2(:,2)).^2)); % RMSE
    test_error(2,4) = median(abs(true_pred_test_2(:,1)-true_pred_test_2(:,2))); % MDAE
    test_error(2,5) = median(abs((true_pred_test_2(:,1)-true_pred_test_2(:,2))./true_pred_test_2(:,1))); % MDAPE
    test_error(2,6) = sqrt(median((true_pred_test_2(:,1)-true_pred_test_2(:,2)).^2)); % RMDSE
end

toc
end