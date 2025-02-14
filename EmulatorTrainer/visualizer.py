#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import arviz as az
# from pymc3 import set_data, sample_posterior_predictive

def plot1d_withCI(x, y, y_std,confidence_level=None):
    lower_bound, upper_bound
    
    y_lower = y-y_std
    

def get_testing_predictions(trace, model, df_testing, varnames):
    dataset = get_model_data(model, df, varnames)
    model.set_data(dataset)
    
def get_model_data(model, df, varnames):
    dataset = {}
    for varname in varnames:
        dataset[varname] = retrieve_combo_variable(varname,df)
    return dataset


def retrieve_combo_variable(varname,df):
    if ':' in varname:
        idx=varname.find(':')
        idx = np.hstack([-1,idx,len(varname)])
        vardat = np.ones(len(df.index))
        for ii in range(len(idx)-1):
            subvar_name = varname[(1+idx[ii]):idx[ii+1]]
            vardat *= retrieve_sub_variable(subvar_name,df)
    else:
        vardat = retrieve_sub_variable(varname,df)
    return vardat
    
    
def retrieve_sub_variable(subvar_name, df):
    matching_vars = [onevar for onevar in df.keys() if onevar in subvar_name]
    varlengths = [len(onevar) for onevar in matching_vars]
    df_varname_list = [
        varname for (varname,varlength) in zip(matching_vars,varlengths) 
        if varlength==max(varlengths)]
    if len(df_varname_list)>1:
        print('error! more than one varname has been matched')
    else:
        df_varname = df_varname_list[0]
    
    x = df[df_varname]
    generalized_subvar_str = subvar_name.replace(df_varname,'x')
    
    return eval(generalized_subvar_str)
    

def get_nmae(y_true, y_pred):
    """
    Compute Normalized Mean Absolute Error (NMAE).

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: NMAE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    norm_factor = np.mean(y_true)  # Normalization by mean of true values
    
    return mae / norm_factor if norm_factor != 0 else np.nan
    
def get_CI_from_data(data,confidence_level,weights=None):
    # I think I need a way to do this as a function of x
    
    idx,=np.argsort(data)
    data_sorted=data[idx]
    weights_sorted=weights[idx]
    
    lower_bound, upper_bound = np.interp(
        np.array([confidence_level/2.,1.-confidence_level/2.]),
        np.cumsum(weights_sorted)/np.sum(weights_sorted),
        data_sorted)
    return lower_bound, upper_bound
    
def get_CI_from_stats(mean, std_dev, confidence_level, sample_size=None):
    
    if sample_size == None:
        Z = get_z_score(confidence_level)
        margin_of_error = Z*std_dev
    else:
        # Get t-score for confidence level
        t_score = st.t.ppf((1 + confidence_level) / 2, df=sample_size-1)
        
        # Compute margin of error
        margin_of_error = t_score * (std_dev / np.sqrt(sample_size))
    
    # Confidence Interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return lower_bound, upper_bound

def get_z_score(confidence_level):
    """
    Returns the Z-score for a given confidence level.
    
    Parameters:
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        float: Z-score corresponding to the confidence level
    """
    alpha = 1 - confidence_level  # Significance level
    z_score = st.norm.ppf(1 - alpha / 2)  # Get critical value from standard normal distribution
    return z_score
