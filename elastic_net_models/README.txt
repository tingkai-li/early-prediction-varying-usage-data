This folder contains the code for predicting lifetime using different sets of features, except the hierarchical linear model which is in the folder "hbm_related_codes".

Two main codes generate results reported in Table 2:
1. "elastic_net_main.m": Generates results for "Dummy Model", "Cycling Condition", "Degradation-informed" with 2 features, and "Degradation-informed" with 3 features.
2. "elastic_net_discharge_model.m": Generates the result for "Discharge Model".

Three user-defined functions are for easier implementation of repeated tasks:
1. "fit_dummy.m": output mean lifetime of the training set
2. "fit_elastic_fixed_train_2_test.m": a customized function to build a cross-validated elastic model and predict on two test sets (i.e., high-DoD and low-DoD in this paper).
3. "plot_pred_results_fixed_in_out.m": a customized function to visualize prediction errors.