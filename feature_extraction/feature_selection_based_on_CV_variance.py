# -*- coding: utf-8 -*-
"""
Code for the feature selection approach included in article "Predicting battery
lifetime from early aging data". 

This code should be run after all features were extracted 
(i.e.,after running "feature_extraction.py").
"""

### Forward feature selection based on CV error variation ###

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score,RepeatedKFold
from sklearn import linear_model

### Define a function to perform feature selection
def feature_selection_mean(x_train_df,y_train,cv_num,cv_repeat, max_feature_num,scoring):
    # Create a list of all features
    all_features = x_train_df.columns.tolist()
    # Create a list to track the added feature for each step
    selected_features = []
    # Create a list to track the remaining features for each step
    remaining_features = all_features.copy()
    # Create dictionaries to track the completed feature subsets and errors
    selected_features_all = {}
    error_all = {}
    run_error_all = {}
    model_fit_check = {}
    
    # Start looping through remaining features, up to the maximum # of features
    for iii in range(max_feature_num):
        run_error=[]
        run_error_var = []
        run_error_mean = []
        for jjj,feature in enumerate(remaining_features):
            feature_eval = selected_features+[feature]
            model = linear_model.LinearRegression()
            x = x_train_df[feature_eval].to_numpy()
            cv_ = RepeatedKFold(n_splits = cv_num,n_repeats = cv_repeat,random_state=0) # Fix random state for reproducibility

            run_error.append(cross_val_score(model, x, y_train,cv=cv_,scoring=scoring))
        for errors in run_error:
            run_error_var.append(errors.std())
            run_error_mean.append(errors.mean())
        
        # Find the feature subset with minimum error of this step
        # Note that the error from regressor is negative RMSE
        idx_min = run_error_mean.index(max(run_error_mean))
        
        # Append error to the dictionary, with the key representing # of features
        run_error_all[iii+1] = run_error[idx_min]
        selected_features.append(remaining_features[idx_min]) # Added the selected feature to the selected feature subset
        remaining_features.pop(idx_min) # Romove the selected feature from candidate pool
    
        selected_features_all[iii+1] = selected_features.copy()
        error_all[iii+1] = [abs(run_error_mean[idx_min]),run_error_var[idx_min]]
        
        model_check = linear_model.LinearRegression().fit(x_train_df[selected_features],y_train)
        pred_check = model_check.predict(x_train_df[selected_features])
        model_fit_check[iii+1]=mean_squared_error(np.exp(y_train), np.exp(pred_check),squared=False)
    return selected_features_all,error_all,run_error_all,model_fit_check

# Import the training data
data = pd.read_csv('training.csv')
data = data.drop(['Group', 'Cell'], axis=1) # Group/cell info is not needed

y_train = np.log(data['Lifetime']) # Tranform to log(lifetime)

x_train_df = data.drop('Lifetime',axis=1) # Keep only features as x

# Select the columns we do not want to apply the log() transform to
columns_to_ignore = ['Chg C-rate', 'Dchg C-rate', 'DoD', 'delta_Q_DVA1', 'delta_Q_DVA2', 
                     'delta_Q_DVA3', 'delta_Q_DVA4','chg_stress', 'dchg_stress', 
                     'avg_stress', 'multi_stress']
df1 = x_train_df[columns_to_ignore]
df1_keys = df1.keys().to_list()
df2 = x_train_df.drop(columns_to_ignore, axis=1)
df2_keys = df2.keys().to_list()

# Log all the columns except the ones we removed
df1 = df1.to_numpy()
df2 = df2.to_numpy()
df2 = np.log(np.abs(df2))

# Combine the data back together
df3 = np.concatenate((df2, df1), axis=1)

# Scale the data
scaler = StandardScaler()
data_norm = scaler.fit_transform(df3)

# Recombine and label the columns
keys = df2_keys + df1_keys
x_train_norm = pd.DataFrame(data_norm, columns=keys)

# Choose the scoring metric for the feature selection
scoring = 'neg_root_mean_squared_error'

# Run the step-wise forward search, up to 10 features
selection,error,run_error_all,model_fit_check = feature_selection_mean(x_train_norm, y_train, 5, 5 ,10,scoring)
model_fit_check_table = pd.DataFrame.from_dict(model_fit_check,orient='index')
all_error_table = pd.DataFrame.from_dict(run_error_all,orient='index')

all_mean = np.abs(np.mean(all_error_table,axis=1))
all_std = np.std(all_error_table,axis=1)

## Visualize the selection result
fontsize = 10

fig, ax1 = plt.subplots(figsize=(3.5,3.5),dpi=600)
plt.rcParams["font.family"] = "Avenir"
color = '#CB4335'
ax1.set_xlabel('Number of Features',fontsize=fontsize)
ax1.set_ylabel('Mean RMSE$_{\mathrm{log(EOL)}}$ [weeks]', color=color,fontsize=fontsize)
ax1.plot(all_mean.index, all_mean, color=color,linewidth=2,marker="o"  )
ax1.tick_params(axis='y', labelcolor=color,labelsize=fontsize)
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.set_xticks([1,2,3,4,5,6,7,8,9,10],labels=['',2,'',4,'',6,'',8,'',10])

plt.grid()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = '#2980B9'
ax2.set_ylabel('RMSE$_{\mathrm{log(EOL)}}$ Standard Deviation [weeks]', color=color,fontsize=fontsize)  # we already handled the x-label with ax1
ax2.plot(all_std.index,  all_std, color=color,linewidth=2,marker="o")
ax2.tick_params(axis='y', labelcolor=color,labelsize=fontsize)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()