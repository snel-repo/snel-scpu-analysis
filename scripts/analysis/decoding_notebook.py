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
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



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

def linear_regression_train_val(x, y, alpha=0):
    """no cross validation"""
    lr = Ridge(alpha=alpha)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   
    lr.fit(x_train, y_train)
    test_pred = lr.predict(x_test)
    train_pred = lr.predict(x_train)

    r2_test = r2_score(y_test, test_pred)
    r2_train = r2_score(y_train, train_pred)
    return test_pred, r2_test, r2_train

# %%
"""
Takes column name and returns predicted kinematic data, true kinematic data, and r2 score
"""
def plot_r2_and_predicted_vs_actual(column_name:str, start_idx: int, stop_idx: int, alpha, use_smooth_data: bool = False):

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
    predicted_vel, r2_test, r2_train = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha)
    return predicted_vel, regression_vel_slice, r2_test, r2_train

fig, axs = plt.subplots(5, 2, figsize=(10, 12))
fig.suptitle('Kinematic Data')
big_ax = fig.add_subplot(111, frameon=False)
# Hide tick and tick label of the big subplot
big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
big_ax.grid(False)

# Set the labels
big_ax.set_xlabel('Time (bins)', labelpad=5)
big_ax.set_ylabel('Velocity', labelpad=0)

ax = axs.flatten()
alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1]

best_alpha = None
best_r2 = -np.inf
r2_values_test = []
r2_values_train = []
r2_sklearn = 0
for alpha in alpha_values:
    mean_r2 = 0
    for idx, column_name in enumerate(dataset.data.kin_pos.columns):
        if idx > 0:
            break
        predicted_vel, vel, r2_test, r2_train = plot_r2_and_predicted_vs_actual(column_name, start_idx=9500+150, stop_idx=9500+750, alpha=alpha, use_smooth_data=False)
        r2_values_test.append(r2_test)
        r2_values_train.append(r2_train)
        
        mean_r2 += r2_sklearn
        ax[idx].plot(vel, label='True')
        ax[idx].plot(predicted_vel, label='Predicted')
        title = f"{column_name}, r^2: {r2_sklearn}"

        ax[idx].set_title(title)
        ax[idx].legend()
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['top'].set_visible(False)
    mean_r2 /= len(dataset.data.kin_pos.columns)

    if mean_r2 > best_r2:
        best_r2 = mean_r2
        best_alpha = alpha
    fig.subplots_adjust(hspace=0.5)
    plt.show()


# %% 
plt.plot(alpha_values, r2_values_test, 'o-', label='Test')
plt.plot(alpha_values, r2_values_train, 'o-', label='Train')

plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.title('R^2 vs Alpha')

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.xlim([0, 1e3])
plt.legend()
plt.show()

# %%
