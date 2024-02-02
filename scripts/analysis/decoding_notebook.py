"""
PURPOSE: Decode the latent factors using a linear decoder

REQUIREMENTS: merged pkl file from merge_chopped_torch_outputs.py
"""

# %% INPUTS AND PATHS
import os
import pickle as pkl
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from snel_toolkit.datasets.nwb import NWBDataset
import logging
import sys
import yaml
import dill
from analysis_utils import *
import scipy.signal as signal
# decoding imports
from snel_toolkit.decoding import prepare_decoding_data
from snel_toolkit.decoding import NeuralDecoder
from sklearn.linear_model import Ridge
import typing
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# %%
# Load everything
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)

dataset, bin_size = load_dataset_and_binsize(yaml_config_path)
# dataset has kinematic data, smoothed spikes, and smoothed lfads_factors

# %% 
# Smooth kinematic data
dataset.smooth_spk(signal_type='kin_pos', gauss_width=8, name='smooth_8', overwrite=False)

# %%
'''
This function takes a dataframe column and returns indices where data is not NaN

Parameters:
    dataframe_column: pandas dataframe column

Returns:
    trials: list of tuples where each tuple is a start and end index of where data is not NaN
'''
def get_change_indices(dataframe_column: pd.Series) -> typing.List[typing.Tuple[int, int]]:
    # Find NaNs
    is_nan = dataframe_column.isna()

    # Find Changes in State
    start_indices = np.where(~is_nan & is_nan.shift(1, fill_value=True))[0]
    end_indices = np.where(is_nan & ~is_nan.shift(1, fill_value=True))[0] - 1

    # Extract Indices
    # +1 to end_indices because range is exclusive
    trials = list(zip(start_indices, end_indices + 1))
    return trials

'''
This function takes a dataframe and indices where data is not NaN and concatenates the data

Parameters:
    df: dataframe of kinematic data
    trials: list of tuples where each tuple is a start and end index of where data is not NaN

Returns:
    concat_trials: numpy array of concatenated non-NaN data
'''

def concatenate_trials(df: pd.DataFrame, trials: typing.List[typing.Tuple[int, int]]) -> np.ndarray:
    # Concatenate trials
    concat_trials = np.concatenate([df.iloc[start:end].values for start, end in trials])
    return concat_trials

# %%

"""
Returns a slice of kinematic and lfads rates data for a given column name

"""
def return_all_nonNan_slice(column_name: str, use_smooth_data: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:

    
    
    if use_smooth_data:
        all_kin_data = dataset.data.kin_pos_smooth_8
        all_rates_data = dataset.data.lfads_rates_smooth_8
    else:
        all_kin_data = dataset.data.kin_pos
        all_rates_data = dataset.data.lfads_rates
    change_indices = get_change_indices(all_kin_data[column_name])
    kin_slice = concatenate_trials(all_kin_data[column_name], change_indices)
    rates_slice = concatenate_trials(all_rates_data, change_indices)
    return kin_slice, rates_slice
"""
Predict kinematic data from lfads rates

Returns predicted kinematic data, linear regression coefficients, and r2 score

"""
def cross_pred(source, target, alpha=1e-2, kfolds=5):
    # source is 2D array of shape (n_bins, n_channels)
    # target is 1D array of shape (n_bins,) where each value is kinematic data
    

    kf = KFold(n_splits=kfolds)
    lr = Ridge(alpha=alpha)
    # pred is 1D array of shape (n_bins,) where each value is predicted kinematic data
    pred = np.zeros_like(target)
    
    lr_coefs = []
    # Loop through kfolds and fit model
    for train_ix, test_ix in kf.split(source):
        X_train, X_test = source[train_ix,:], source[test_ix,:]
        y_train, y_test = target[train_ix], target[test_ix]
        lr.fit(X_train, y_train)
        lr_coefs.append(lr.coef_)
        pred[test_ix] = lr.predict(X_test)



    lr_coef = np.mean(np.stack(lr_coefs),axis=0)  
    r2 = r2_score(target, pred)
    return pred, lr_coef, r2



# %% 

def diff_filter(x):
        """differentation filter"""
        return signal.savgol_filter(x, 27, 5, deriv=1, axis=0)

def linear_regression(x, y):
    """With GridSearchCV"""
    lr = Ridge()
    
    # Define the grid of hyperparameters to search
    param_grid = {'alpha': [0, 0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Set up the grid search
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
    
    # Perform the grid search
    grid_search.fit(x, y)
    
    # Use the best model to make predictions
    pred = grid_search.best_estimator_.predict(x)
    
    # Get the best R^2 score and the best parameters
    r2 = grid_search.best_score_
    alpha = grid_search.best_params_["alpha"]
    
    return pred, r2, alpha

# %%
"""
Takes column name and returns predicted kinematic data, true kinematic data, and r2 score
"""
def predict_kinematics(column_name:str, start_idx: int, stop_idx: int, use_smooth_data: bool = False):

    kin_slice, rates_slice = return_all_nonNan_slice(column_name, use_smooth_data=use_smooth_data)
    vel = diff_filter(kin_slice)


    # slice vel and rates_slice to exclude handpicked_outliers

    regression_vel_slice = vel[start_idx:stop_idx]
    regression_rates_slice = rates_slice[start_idx:stop_idx]

    # determine outliers    
    outlier_min = np.argmin(regression_vel_slice)
    outlier_max = np.argmax(regression_vel_slice)



    # remove outliers
    regression_vel_slice = np.delete(regression_vel_slice, [outlier_min, outlier_max])
    regression_rates_slice = np.delete(regression_rates_slice, [outlier_min, outlier_max], axis=0)

    # Predict kinematic data from lfads rates
    #predicted_vel, _, r2_sklearn = cross_pred(regression_rates_slice, regression_vel_slice, alpha=alpha, kfolds=10)
    predicted_vel, r2_sklearn, best_alpha = linear_regression(regression_rates_slice, regression_vel_slice)
    return predicted_vel, regression_vel_slice, r2_sklearn, best_alpha

fig, axs = plt.subplots(5, 2, figsize=(10, 12))
fig.suptitle('Cross validated linear regression predictions of kinematic data from lfads rates', fontsize=16)
big_ax = fig.add_subplot(111, frameon=False)
# Hide tick and tick label of the big subplot
big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
big_ax.grid(False)

# Set the labels
big_ax.set_xlabel('Time (bins)', labelpad=5)
big_ax.set_ylabel('Velocity', labelpad=0)

ax = axs.flatten()

best_alpha = None
best_r2 = -np.inf
best_alphas = []
r2_values_per_column = np.zeros(len(dataset.data.kin_pos.columns))
for idx, column_name in enumerate(dataset.data.kin_pos.columns):
    predicted_vel, vel, r2_sklearn, best_alpha = predict_kinematics(column_name, start_idx=9500+150, stop_idx=9500+750, use_smooth_data=False)
    r2_values_per_column[idx] = r2_sklearn
    best_alphas.append(best_alpha)
    
    ax[idx].plot(vel, label='True')
    ax[idx].plot(predicted_vel, label='Predicted')
    title = f"{column_name}, r^2: {r2_sklearn}"

    ax[idx].set_title(title)
    ax[idx].legend()
    ax[idx].spines['right'].set_visible(False)
    ax[idx].spines['top'].set_visible(False)

fig.subplots_adjust(hspace=0.5)
plt.show()

# %% 


# plot ankle_x position and velocity over time

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
kin_slice, rates_slice = return_all_nonNan_slice('ankle_x', use_smooth_data=False)
vel = diff_filter(kin_slice)
ax[0].plot(kin_slice)
ax[0].set_title('ankle_x position over time')
ax[0].set_xlabel('Time (bins)')
ax[0].set_ylabel('Position')
ax[0].legend()
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

ax[1].plot(vel)
ax[1].set_title('ankle_x velocity over time')
ax[1].set_xlabel('Time (bins)')
ax[1].set_ylabel('Velocity')
ax[1].legend()
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
fig.subplots_adjust(hspace=0.5)
plt.show()

# %%
