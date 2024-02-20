# -*- coding: utf-8 -*-
"""

This code is for extracting equivalent "discharge model" features from the paper
"Data-driven prediction of battery cycle life before capacity degradation"
by Severson et al.

"""
import json
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.stats import kurtosis
from scipy.stats import skew

### Define a function to calculate the lifetime ###
def lifetime(capacity_fade_dir,cell,subfolder):
    # Read capacity fade from preprocessed CSV file
    capacity_fade_data = pd.read_csv(capacity_fade_dir+subfolder+cell+'.csv')
    # Raw measurements (Q and time)
    xi = capacity_fade_data['Time'].values
    yi = capacity_fade_data['Capacity'].values
    # Interpolation range of time
    x = np.arange(0,np.ceil(np.max(xi)),0.001)
    # Interpolated capacity fade
    y = pchip_interpolate(xi, yi, x)
    
    # Find the lifetime (i.e., time stamp at which y <=0.2)
    life_idx = np.argmin(np.abs(y-0.2))
    life = x[life_idx]
    return np.round(life,3)

### Define a function for find initial capacity ###
def initial_Q(capacity_fade_dir,cell,subfolder):
    # Read capacity fade from preprocessed CSV file
    capacity_fade_data = pd.read_csv(capacity_fade_dir+subfolder+cell+'.csv')
    return capacity_fade_data['Capacity'].values[0]
 
### Define a function to extract equivalent discharge features ### 
def delta_QV_features(QV_int_dir,cell,subfolder):
    # Transpose the preprocessed interpolated QV curves to have each row 
    # representing a week, and save as np array for easier indexing
    QV_discharge_interpolate = pd.read_csv(QV_int_dir+subfolder+cell+'.csv',header=None).T
    QV_array = QV_discharge_interpolate.to_numpy()
    
    g = int(cell[1:-2])  # Group number
    if g < 20:
        delta_Q = QV_array[3] - QV_array[0]
    else:
        delta_Q = QV_array[4] - QV_array[0]
    
    return np.var(delta_Q),np.min(delta_Q),skew(delta_Q),kurtosis(delta_Q)



if __name__ == '__main__':
    # List of cells used in this paper
    valid_cells = pd.read_csv('valid_cells_paper.csv').values.flatten().tolist()
    
    # Cell in Release 2.0 (to determine the proper subfolder in released dataset)
    batch2 = ['G57C1','G57C2','G57C3','G57C4','G58C1', 'G26C3','G49C1','G49C2','G49C3','G49C4','G50C1','G50C3','G50C4'] 
    
    
    ### Main directory for the dataset ### TO MODIFY
    main_dir = ''
    # Directory of raw JSON files for RPT
    RPT_json_dir = main_dir+'/RPT_json/'
    # Directory of preprocessed (interpolated) QV curves
    QV_int_dir = main_dir+'/Q_interpolated/'
    # Directory of preprocessed capacity vs. time data
    capacity_fade_dir = main_dir+'/capacity_fade/'
    


    # Create a pandas data frame for storing all features
    feature_df = pd.DataFrame()
    for i,cell in enumerate(valid_cells):
        if cell in batch2:
            subfolder = 'Release 2.0/'
        else:
            subfolder = 'Release 1.0/'
        
        feature_df.loc[i,'Group'] = cell[0:-2]
        feature_df.loc[i,'Cell'] = cell
        
        # Calculate the lifetime
        feature_df.loc[i,'Lifetime'] = lifetime(capacity_fade_dir,cell,subfolder)
        # Find features
        feature_df.loc[i,'var_deltaQ'],feature_df.loc[i,'min_deltaQ'], \
        feature_df.loc[i,'skew_deltaQ'],feature_df.loc[i,'kurt_deltaQ'] = \
        delta_QV_features(QV_int_dir,cell,subfolder)
        
        feature_df.loc[i,'Q_ini'] = initial_Q(capacity_fade_dir, cell, subfolder)
        
    
        
    # Train/test split based on groups
    training = ['G10', 'G14', 'G16', 'G19', 'G2', 'G20', 'G22', 'G23', 'G25', 
                'G27', 'G28', 'G3', 'G30', 'G31', 'G35', 'G36', 'G38', 'G4', 
                'G41', 'G45', 'G47', 'G48', 'G5', 'G51', 'G52', 'G53', 'G55', 
                'G60', 'G62', 'G7']

    test_in = ['G13', 'G17', 'G21', 'G24', 'G29', 'G33', 'G37', 'G39', 'G46', 
               'G56', 'G58', 'G61', 'G63', 'G64', 'G8', 'G9']
    test_out = ['G1', 'G12', 'G18', 'G26', 'G32', 'G34', 'G40', 'G42', 'G43', 
                'G44', 'G50', 'G54', 'G59', 'G6']
    
    # Partition the data into three subsets based on groups
    feature_df_training = feature_df.loc[feature_df['Group'].isin(training)]
    feature_df_test_in  = feature_df.loc[feature_df['Group'].isin(test_in)]
    feature_df_test_out = feature_df.loc[feature_df['Group'].isin(test_out)]
    
    # Save data as CSV files for feature selection and machine learning
    feature_df.to_csv('feature_discharge_all.csv',index=False)
    feature_df_training.to_csv('training_discharge.csv',index=False)
    feature_df_test_in.to_csv('test_in_discharge.csv',index=False)
    feature_df_test_out.to_csv('test_out_discharge.csv',index=False)