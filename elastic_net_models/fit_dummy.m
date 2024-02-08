function [training_error, test_error,  true_pred_test] = fit_dummy(Y_train, Y_test, log_target)

% Lay out the default values for the method
arguments
    Y_train (:,1) {double}
    Y_test  (:,1) {double}
    log_target = true
end

training_error = zeros(1,6);
test_error = zeros(1,6);

mean_train = mean(Y_train);
y_hat_test = ones(height(Y_test),1).*mean_train;
y_hat_train = ones(height(Y_train),1).*mean_train;

% Calculate error (MAE,MAPE,MSE,MDAE,MDAPE,MDSE), MD=median
if log_target == true
    % Calculate training error
    y_pred_train = exp(y_hat_train);
    y_true_train = exp(Y_train);
    true_pred_train = [y_true_train y_pred_train];

    training_error(1,1) = mean(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MAE
    training_error(1,2) = mean(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MAPE
    training_error(1,3) = sqrt(mean((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMSE
    training_error(1,4) = median(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MDAE
    training_error(1,5) = median(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MDAPE
    training_error(1,6) = sqrt(median((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMDSE

    % Calculate test error
    y_pred_test = exp(y_hat_test);
    y_true_test = exp(Y_test);
    true_pred_test = [y_true_test y_pred_test];

    test_error(1,1) = mean(abs(true_pred_test(:,1)-true_pred_test(:,2))); % MAE
    test_error(1,2) = mean(abs((true_pred_test(:,1)-true_pred_test(:,2))./true_pred_test(:,1)))*100; % MAPE
    test_error(1,3) = sqrt(mean((true_pred_test(:,1)-true_pred_test(:,2)).^2)); % RMSE
    test_error(1,4) = median(abs(true_pred_test(:,1)-true_pred_test(:,2))); % MDAE
    test_error(1,5) = median(abs((true_pred_test(:,1)-true_pred_test(:,2))./true_pred_test(:,1))); % MDAPE
    test_error(1,6) = sqrt(median((true_pred_test(:,1)-true_pred_test(:,2)).^2)); % RMDSE
elseif log_target == false
    % Calculate training error
    true_pred_train = [Y_train y_hat_train];

    training_error(1,1) = mean(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MAE
    training_error(1,2) = mean(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MAPE
    training_error(1,3) = sqrt(mean((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMSE
    training_error(1,4) = median(abs(true_pred_train(:,1)-true_pred_train(:,2))); % MDAE
    training_error(1,5) = median(abs((true_pred_train(:,1)-true_pred_train(:,2))./true_pred_train(:,1))); % MDAPE
    training_error(1,6) = sqrt(median((true_pred_train(:,1)-true_pred_train(:,2)).^2)); % RMDSE

    % Calculate test error
    true_pred_test = [Y_test y_hat_test];

    test_error(1,1) = mean(abs(true_pred_test(:,1)-true_pred_test(:,2))); % MAE
    test_error(1,2) = mean(abs((true_pred_test(:,1)-true_pred_test(:,2))./true_pred_test(:,1)))*100; % MAPE
    test_error(1,3) = sqrt(mean((true_pred_test(:,1)-true_pred_test(:,2)).^2)); % RMSE
    test_error(1,4) = median(abs(true_pred_test(:,1)-true_pred_test(:,2))); % MDAE
    test_error(1,5) = median(abs((true_pred_test(:,1)-true_pred_test(:,2))./true_pred_test(:,1))); % MDAPE
    test_error(1,6) = sqrt(median((true_pred_test(:,1)-true_pred_test(:,2)).^2)); % RMDSE
end

end