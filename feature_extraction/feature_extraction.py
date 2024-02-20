# -*- coding: utf-8 -*-
"""

Code for extracting early-life features included in article "Predicting battery
lifetime from early aging data". 

ISU-ILCC battery aging dataset is available at:
https://doi.org/10.25380/iastate.22582234
For more details about the structure of raw data, please refer to the documentation
of the dataset.

This code is built on:
python - 3.7.13 (Python 3.9.9 has been verified to be compatible)
numpy - 1.21.5
scipy - 1.7.3
pandas - 1.3.5

"""
### Import required packages ###
import json
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

### Define a function to load RPT data and convert the data type ###
def convert_RPT_to_dict(RPT_json_dir,cell,subfolder):
    with open(RPT_json_dir+subfolder+cell+'.json','r') as file:
        data_dict = json.loads(json.load(file))
    
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['start_stop_time']['start'])):
        data_dict['QV_charge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_2']['t'][iii]))
        data_dict['QV_discharge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_2']['t'][iii]))
        data_dict['QV_charge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_5']['t'][iii]))
        data_dict['QV_discharge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_5']['t'][iii]))
    
    # Return the preprocessed Python Dictionary
    return data_dict

### Define a function to differentiate interpolated QV curves
def diff_QV(QV_int_dir,cell,subfolder,V_interpolate):
    # Transpose the preprocessed interpolated QV curves to have each row 
    # representing a week, and save as np array for easier indexing
    QV_discharge_interpolate = pd.read_csv(QV_int_dir+subfolder+cell+'.csv',header=None).T
    QV_array = QV_discharge_interpolate.to_numpy()
    
    # Create empty lists and find numerical derivatives of the QV curve
    dQdV = []; dVdQ = []
    for i in range(len(QV_array)):
        Q = QV_array[i]
        dQdV.append(np.diff(Q)/np.diff(V_interpolate))
        dVdQ.append(np.diff(V_interpolate)/np.diff(Q))
    
    # Save and return as arrays for easier indexing afterward
    dQdV_array = np.array(dQdV)
    dVdQ_array = np.array(dVdQ)
    return QV_array, dQdV_array, dVdQ_array

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
    
### Define a function to extract curve differences within a given voltage window ###
def curve_difference_varying_window(cell, vlow, vhigh, vmin, vmax, vlength, start_week_num, end_week_num, interpolated_curve):
    interpolate_x = np.linspace(vmin,vmax,vlength)
    idx_window = (interpolate_x >= vlow) * (interpolate_x <= vhigh)

    # Groups of cells beyond G20 had an additional RPT at week 0.5
    g = int(cell[1:-2])
    if g<20:
        try:
            curve_diff = interpolated_curve[cell][end_week_num][idx_window] - interpolated_curve[cell][start_week_num][idx_window]
        except:
            curve_diff = np.nan

    else:
        try:
            if start_week_num==0:
                curve_diff = interpolated_curve[cell][end_week_num+1][idx_window] - interpolated_curve[cell][start_week_num][idx_window]
            else:
                curve_diff = interpolated_curve[cell][end_week_num+1][idx_window] - interpolated_curve[cell][start_week_num+1][idx_window]
        except:
            curve_diff = np.nan
    
    return curve_diff

### Define a function to extract features from dVdQ curves ###
def DVA_capacity_features(cell, week1, week2, dvdq_curves, Q_int):
    # We use an upper plot limit of 6.0 for the dVdQ curves
    y_lim = 6
  
    g = int(cell[1:-2])  # Group number
    if g < 20:
        # Shortening the curves makes it easier to locate peaks
        idxs1_ = (Q_int[cell][week1][:-1] <= 0.28) * (-dvdq_curves[cell][week1] <= y_lim) * (dvdq_curves[cell][week1] <= 0)
        idxs2_ = (Q_int[cell][week2][:-1] <= 0.28) * (-dvdq_curves[cell][week2] <= y_lim) * (dvdq_curves[cell][week2] <= 0)
        q1_ = Q_int[cell][week1][:-1][idxs1_]
        q2_ = Q_int[cell][week2][:-1][idxs2_]
        dvdq1_ = -dvdq_curves[cell][week1][idxs1_]
        dvdq2_ = -dvdq_curves[cell][week2][idxs2_]
        
        # Reverse Q and dVdQ curve such that Q is ascending
        q1_ = q1_[::-1]
        dvdq1_ = dvdq1_[::-1]
        q2_ = q2_[::-1]
        dvdq2_ = dvdq2_[::-1]

        # Find local maximums
        max_idxs1 = argrelextrema(dvdq1_, np.greater, axis=0, order=10)[0]
        max_idxs2 = argrelextrema(dvdq2_, np.greater, axis=0, order=10)[0]

        # find the right peak of p1
        QNE1_ind = max_idxs1[1]
        for ind in max_idxs1:
            if q1_[ind] < 0.12 or q1_[ind] > 0.18:
                continue
            if dvdq1_[ind] > dvdq1_[QNE1_ind]:
                QNE1_ind = ind

        # debug for merged final peak in dvdq1_
        if q1_[max_idxs1[-1]] < 0.19:
            min_ind = argrelextrema(dvdq1_[max_idxs1[1]:], np.less, axis=0, order=10)[0]
            diff_dvdq1 = np.diff(dvdq1_[max_idxs1[1]:])
            zero_diff_ind = np.where(np.logical_and(diff_dvdq1<0.006,diff_dvdq1>-0.006))[0]

            if zero_diff_ind[-1] > min_ind[-1]+10:
                max_idxs1 = np.append(max_idxs1,max_idxs1[1]+zero_diff_ind[-1])
            else:
                correct_zero_diff = zero_diff_ind[np.where(zero_diff_ind < min_ind[-1]-10)]
                max_idxs1 = np.append(max_idxs1,max_idxs1[1]+correct_zero_diff[-1])

        QNE2_ind = max_idxs2[1]
        for ind in max_idxs2:
            if q2_[ind] < 0.12 or q2_[ind] > 0.18:
                continue
            if dvdq2_[ind] > dvdq2_[QNE2_ind]:
                QNE2_ind = ind

        # debug for merged final peak in dvdq2_
        if q2_[max_idxs2[-1]] < 0.19:
            min_ind = argrelextrema(dvdq2_[max_idxs2[1]:], np.less, axis=0, order=10)[0]
            diff_dvdq2 = np.diff(dvdq2_[max_idxs2[1]:])
            zero_diff_ind = np.where(np.logical_and(diff_dvdq2<0.006,diff_dvdq2>-0.006))[0]

            if zero_diff_ind[-1] > min_ind[-1]+10:
                max_idxs2 = np.append(max_idxs2,max_idxs2[1]+zero_diff_ind[-1])
            else:
                correct_zero_diff = zero_diff_ind[np.where(zero_diff_ind < min_ind[-1]-10)]
                max_idxs2 = np.append(max_idxs2,max_idxs2[1]+correct_zero_diff[-1])

        delta_Q1 = (q1_[-1] - q1_[max_idxs1[-1]]) - (q2_[-1] - q2_[max_idxs2[-1]])
        delta_Q2 = (q1_[QNE1_ind] - q1_[0]) - (q2_[QNE2_ind] - q2_[0])
        delta_Q3 = (q1_[-1] - q1_[QNE1_ind]) - (q2_[-1] - q2_[QNE2_ind])
        delta_Q4 = (q1_[max_idxs1[-1]] - q1_[QNE1_ind]) - (q2_[max_idxs2[-1]] - q2_[QNE2_ind])
        delta_Q = (q1_[-1] - q1_[0]) - (q2_[-1] - q2_[0])
        
    else:
        # Shortening the curves makes it easier to locate peaks
        idxs1_ = (Q_int[cell][week1][:-1] <= 0.28) * (-dvdq_curves[cell][week1] <= y_lim) * (dvdq_curves[cell][week1] <= 0)
        idxs2_ = (Q_int[cell][week2+1][:-1] <= 0.28) * (-dvdq_curves[cell][week2+1] <= y_lim) * (dvdq_curves[cell][week2+1] <= 0)

        q1_ = Q_int[cell][week1][:-1][idxs1_]
        q2_ = Q_int[cell][week2+1][:-1][idxs2_]
        dvdq1_ = -dvdq_curves[cell][week1][idxs1_]
        dvdq2_ = -dvdq_curves[cell][week2+1][idxs2_]
        
        # Reverse Q and dVdQ curve such that Q is ascending
        q1_ = q1_[::-1]
        dvdq1_ = dvdq1_[::-1]
        q2_ = q2_[::-1]
        dvdq2_ = dvdq2_[::-1]

        # Find local maximums
        max_idxs1 = argrelextrema(dvdq1_, np.greater, axis=0, order=10)[0]
        max_idxs2 = argrelextrema(dvdq2_, np.greater, axis=0, order=10)[0]

        # find the right peak of p2
        QNE1_ind = max_idxs1[1]
        for ind in max_idxs1:
            if q1_[ind] < 0.12 or q1_[ind] > 0.18:
                continue
            if dvdq1_[ind] > dvdq1_[QNE1_ind]:
                QNE1_ind = ind

        # debug for merged final peak in dvdq1_
        if q1_[max_idxs1[-1]] < 0.19:
            min_ind = argrelextrema(dvdq1_[max_idxs1[1]:], np.less, axis=0, order=10)[0]
            diff_dvdq1 = np.diff(dvdq1_[max_idxs1[1]:])
            zero_diff_ind = np.where(np.logical_and(diff_dvdq1<0.006,diff_dvdq1>-0.006))[0]

            if zero_diff_ind[-1] > min_ind[-1]+10:
                max_idxs1 = np.append(max_idxs1,max_idxs1[1]+zero_diff_ind[-1])
            else:
                correct_zero_diff = zero_diff_ind[np.where(zero_diff_ind < min_ind[-1]-10)]
                max_idxs1 = np.append(max_idxs1,max_idxs1[1]+correct_zero_diff[-1])

        QNE2_ind = max_idxs2[1]
        for ind in max_idxs2:
            if q2_[ind] < 0.12 or q2_[ind] > 0.18:
                continue
            if dvdq2_[ind] > dvdq2_[QNE2_ind]:
                QNE2_ind = ind
                
        # debug for merged final peak in dvdq2_
        if q2_[max_idxs2[-1]] < 0.19:
            min_ind = argrelextrema(dvdq2_[max_idxs2[1]:], np.less, axis=0, order=10)[0]
            diff_dvdq2 = np.diff(dvdq2_[max_idxs2[1]:])
            zero_diff_ind = np.where(np.logical_and(diff_dvdq2<0.006,diff_dvdq2>-0.006))[0]

            if zero_diff_ind[-1] > min_ind[-1]+10:
                max_idxs2 = np.append(max_idxs2,max_idxs2[1]+zero_diff_ind[-1])
            else:
                correct_zero_diff = zero_diff_ind[np.where(zero_diff_ind < min_ind[-1] - 10)]
                max_idxs2 = np.append(max_idxs2, max_idxs2[1] + correct_zero_diff[-1])

        delta_Q1 = (q1_[-1] - q1_[max_idxs1[-1]]) - (q2_[-1] - q2_[max_idxs2[-1]])
        delta_Q2 = (q1_[QNE1_ind] - q1_[0]) - (q2_[QNE2_ind] - q2_[0])
        delta_Q3 = (q1_[-1] - q1_[QNE1_ind]) - (q2_[-1] - q2_[QNE2_ind])
        delta_Q4 = (q1_[max_idxs1[-1]] - q1_[QNE1_ind]) - (q2_[max_idxs2[-1]] - q2_[QNE2_ind])
        delta_Q = (q1_[-1] - q1_[0]) - (q2_[-1] - q2_[0])


    return delta_Q1, delta_Q2, delta_Q3, delta_Q4

### Define a function to find CV time ###
def charge_CV_feature(cell, RPT_dict, week_num):     
    g = int(cell[1:-2])  # Group number
    if g < 20:
        try:
            index = np.where(np.array(RPT_dict['QV_charge_C_5']['I'][week_num])<0.0495)[0][0]
            t_CV = (RPT_dict['QV_charge_C_5']['t'][week_num][-1] - RPT_dict['QV_charge_C_5']['t'][week_num][index])/np.timedelta64(1,'s')
        except:
            t_CV = np.nan
    else:
        try:
            if week_num ==0:
                index = np.where(np.array(RPT_dict['QV_charge_C_5']['I'][week_num])<0.0495)[0][0]
                t_CV = (RPT_dict['QV_charge_C_5']['t'][week_num][-1] - RPT_dict['QV_charge_C_5']['t'][week_num][index])/np.timedelta64(1,'s')
            else:
                index = np.where(np.array(RPT_dict['QV_charge_C_5']['I'][week_num+1])<0.0495)[0][0]
                t_CV = (RPT_dict['QV_charge_C_5']['t'][week_num+1][-1] - RPT_dict['QV_charge_C_5']['t'][week_num+1][index])/np.timedelta64(1,'s')
        except:
            t_CV = np.nan
    return t_CV

### Define a function to extract initial capacity and capacity fade features ###
def capacity_features(RPT_dict, cell, end_week):
    Q_initial = RPT_dict['capacity_discharge_C_5'][0]

    g = int(cell[1:-2])  # Group number
    if g < 20:
        Q_fade = RPT_dict['capacity_discharge_C_5'][0] - RPT_dict['capacity_discharge_C_5'][end_week]
    else:
        Q_fade = RPT_dict['capacity_discharge_C_5'][0] - RPT_dict['capacity_discharge_C_5'][end_week+1]

    return Q_initial, Q_fade

### Define a function to extract the initial capacity within a given voltage window ###
def initial_Q_varying_window(cell, vlow, vhigh, vmin, vmax, vlength, interpolated_curve):
    idx_1 =  np.where((np.linspace(vmin, vmax, vlength) >= vlow))[0][0]
    idx_2 =  np.where((np.linspace(vmin, vmax, vlength) >= vhigh))[0][0]
    return interpolated_curve[cell][0][idx_2]-interpolated_curve[cell][0][idx_1]



if __name__ == '__main__':
    # List of cells used in this paper. Some cells from the dataset didn't 
    # reach the EOL at the time when we finalized results for this paper 
    valid_cells = pd.read_csv('valid_cells_paper.csv').values.flatten().tolist()
    
    # Cycling conditions of each group (without DoD) and empirical DoD values for each cell
    cycling_cond = pd.read_csv('cycling_conditions_wo_DoD.csv')
    empirical_DoD = pd.read_csv('empirical_DoD.csv')
    
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
    
    
    # Create dictionaries for QV curves, dQdV curves, and dVdQ curves
    QV_dict = {}; dQdV_dict = {}; dVdQ_dict = {}
    life = {}
    # Create dictionaries for curve differences
    delta_QV = {}; delta_dQdV = {}
    delta_dQdV_low = {}; delta_dQdV_mid = {}; delta_dQdV_high = {}
    # Create dictionaries for DVA features
    Q_dva_1 = {}; Q_dva_2 = {}; Q_dva_3 = {}; Q_dva_4 = {}
    # Create dictionaries for CV time features
    cv_time_0 = {}; cv_time_3 = {}; delta_CV_time = {};
    # Create dictionaries for initial capacity, initial capacity with varying 
    # windows, capacity fade
    Q_initial = {}; Q_initial_low = {}; Q_initial_mid = {}; Q_initial_high = {}
    Q_fade = {}    
    
    # 1000 equally spaced voltage points for deriving derivatives. This is for 
    # consistency with the interpolation during the preprocessing. When we extract
    # features, we use the full voltage range (3.0V to 4.2V).
    V_interpolate = np.linspace(3,4.18,1000)  
    
    for i,cell in enumerate(valid_cells):
        if cell in batch2:
            subfolder = 'Release 2.0/'
        else:
            subfolder = 'Release 1.0/'
        # Calculate the lifetime
        life[cell] = lifetime(capacity_fade_dir,cell,subfolder)
        # Derive differential curves
        QV_dict[cell],dQdV_dict[cell],dVdQ_dict[cell] = diff_QV(QV_int_dir,cell,subfolder,V_interpolate)
        # Find curve differences
        delta_QV[cell] = curve_difference_varying_window(cell, 3, 4.2, 3, 4.2, 1000, 0, 3, QV_dict)
        delta_dQdV[cell] = curve_difference_varying_window(cell, 3, 4.2, 3, 4.2, 999, 0, 3, dQdV_dict)
        delta_dQdV_low[cell] = curve_difference_varying_window(cell, 3, 3.6, 3, 4.2, 999, 0, 3, dQdV_dict)
        delta_dQdV_mid[cell] = curve_difference_varying_window(cell, 3.6, 3.9, 3, 4.2, 999, 0, 3, dQdV_dict)
        delta_dQdV_high[cell] = curve_difference_varying_window(cell, 3.9, 4.2, 3, 4.2, 999, 0, 3, dQdV_dict)
        Q_dva_1[cell],Q_dva_2[cell],Q_dva_3[cell],Q_dva_4[cell] = DVA_capacity_features(cell, 0, 3, dVdQ_dict, QV_dict)
        # Read RPT file
        RPT_dict = convert_RPT_to_dict(RPT_json_dir,cell,subfolder)
        # Find CV time
        cv_time_0[cell] = charge_CV_feature(cell, RPT_dict, 0)
        cv_time_3[cell] = charge_CV_feature(cell, RPT_dict, 3)
        delta_CV_time[cell] = cv_time_3[cell] - cv_time_0[cell]
        # Find initial capacity features
        Q_initial[cell], Q_fade[cell] = capacity_features(RPT_dict, cell, 3)
        Q_initial_low[cell] = initial_Q_varying_window(cell, 3, 3.6, 3, 4.2, 1000, QV_dict)
        Q_initial_mid[cell] = initial_Q_varying_window(cell, 3.6, 3.9, 3, 4.2, 1000, QV_dict)
        Q_initial_high[cell] = initial_Q_varying_window(cell, 3.9, 4.2, 3, 4.2, 1000, QV_dict)

    # Organize extracted features into a pandas dataframe
    feature_df = pd.DataFrame()
    for i,cell in enumerate(valid_cells):
        chg_Crate = cycling_cond[cycling_cond['Group#']==int(cell[1:-2])]['Charging C-rate'].values
        dchg_Crate = cycling_cond[cycling_cond['Group#']==int(cell[1:-2])]['Discharging C-rate'].values
        DoD = empirical_DoD[empirical_DoD['Cell']==cell]['DoD'].values
        chg_stress = np.sqrt(chg_Crate)*np.sqrt(DoD)
        dchg_stress = np.sqrt(dchg_Crate)*np.sqrt(DoD)
        avg_stress = (chg_stress+dchg_stress)/2
        multi_stress = chg_stress * dchg_stress
        feature_df.loc[i,'Group'] = cell[0:-2]
        feature_df.loc[i,'Cell'] = cell
        feature_df.loc[i,'Chg C-rate'] = chg_Crate
        feature_df.loc[i,'Dchg C-rate'] = dchg_Crate
        feature_df.loc[i,'DoD'] = DoD
        feature_df.loc[i,'Lifetime'] = life[cell]
        feature_df.loc[i,'Q_initial'] = Q_initial[cell]
        feature_df.loc[i,'Q_ini_V_low'] = Q_initial_low[cell]
        feature_df.loc[i,'Q_ini_V_mid'] = Q_initial_mid[cell]
        feature_df.loc[i,'Q_ini_V_high'] = Q_initial_high[cell]
        feature_df.loc[i,'CV_time_0'] = cv_time_0[cell]
        feature_df.loc[i,'CV_time_3'] = cv_time_3[cell]
        feature_df.loc[i,'delta_CV_time_3_0'] = delta_CV_time[cell]
        feature_df.loc[i,'capacity_fade_3_0'] = Q_fade[cell]
        feature_df.loc[i,'var_deltaQ_dchg_3_0'] = np.var(delta_QV[cell])
        feature_df.loc[i,'mean_deltaQ_dchg_3_0'] = np.mean(delta_QV[cell])
        feature_df.loc[i,'var_dqdv_dchg_3_0'] = np.var(delta_dQdV[cell])
        feature_df.loc[i,'mean_dqdv_dchg_3_0'] = np.mean(delta_dQdV[cell])
        feature_df.loc[i,'var_dqdv_dchg_low_3_0'] = np.var(delta_dQdV_low[cell])
        feature_df.loc[i,'mean_dqdv_dchg_low_3_0'] = np.mean(delta_dQdV_low[cell])
        feature_df.loc[i,'var_dqdv_dchg_mid_3_0'] = np.var(delta_dQdV_mid[cell])
        feature_df.loc[i,'mean_dqdv_dchg_mid_3_0'] = np.mean(delta_dQdV_mid[cell])
        feature_df.loc[i,'var_dqdv_dchg_high_3_0'] = np.var(delta_dQdV_high[cell])
        feature_df.loc[i,'mean_dqdv_dchg_high_3_0'] = np.mean(delta_dQdV_high[cell])
        feature_df.loc[i,'delta_Q_DVA1'] = Q_dva_1[cell]
        feature_df.loc[i,'delta_Q_DVA2'] = Q_dva_2[cell]
        feature_df.loc[i,'delta_Q_DVA3'] = Q_dva_3[cell]
        feature_df.loc[i,'delta_Q_DVA4'] = Q_dva_4[cell]
        feature_df.loc[i,'chg_stress'] = chg_stress
        feature_df.loc[i,'dchg_stress'] = dchg_stress
        feature_df.loc[i,'avg_stress'] = avg_stress
        feature_df.loc[i,'multi_stress'] = multi_stress
        
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
    feature_df.to_csv('feature_all.csv',index=False)
    feature_df_training.to_csv('training.csv',index=False)
    feature_df_test_in.to_csv('test_in.csv',index=False)
    feature_df_test_out.to_csv('test_out.csv',index=False)