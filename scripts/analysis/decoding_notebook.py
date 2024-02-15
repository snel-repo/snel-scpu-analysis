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
from typing import List
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.cm as cm
import matplotlib.colors as colors




# %%
# Load everything
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)

dataset, bin_size = load_dataset_and_binsize(yaml_config_path)
# dataset has kinematic data, smoothed spikes, and smoothed lfads_factors

# %% 
# Smooth kinematic data and factors
dataset.smooth_spk(signal_type='kin_pos', gauss_width=3, name='smooth_3', overwrite=False)
# dataset.smooth_spk(signal_type='lfads_factors', gauss_width=8, name='smooth_8', overwrite=False)
# dataset.data.lfads_factors_smooth_8 = dataset.data.lfads_factors_smooth_8.fillna(0)

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
Returns all nonNan data of kinematic data for a given column name and returns lfads rates based on those indices

"""
def return_all_nonNan_slice(column_name: str, use_smooth_data: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:

    
    
    if use_smooth_data:
        all_kin_data = dataset.data.kin_pos_smooth_3
        all_rates_data = dataset.data.lfads_rates_smooth_8
        
    else:
        all_kin_data = dataset.data.kin_pos
        all_rates_data = dataset.data.lfads_rates
    change_indices = get_change_indices(all_kin_data[column_name])
    kin_slice = concatenate_trials(all_kin_data[column_name], change_indices)
    rates_slice = concatenate_trials(all_rates_data, change_indices)
    return kin_slice, rates_slice


# %% 

def diff_filter(x):
        """differentation filter"""
        return signal.savgol_filter(x, 27, 5, deriv=1, axis=0)


def linear_regression_train_val(x, y, alpha=0, folds=5):
    """
    Function to perform linear regression with cross validation to decode kinematic data from neural data (eg. lfads rates, factors, smooth spikes)
    x is 2D array of shape (n_bins, n_channels) which is (samples, features)
    y is 1D array of shape (n_bins,) where each value is kinematic data
    """

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    lr = Ridge(alpha=alpha)
    
    r2_test = 0
    r2_train = 0
    fold_mean_x = [] # for test data
    fold_mean_y = [] # for test data
    test_pred = np.zeros_like(y)

    for train_ix, test_ix in kf.split(x):
        x_train, x_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]


        lr.fit(x_train, y_train)
        test_pred[test_ix] = lr.predict(x_test)
        train_pred = lr.predict(x_train)
        r2_test += r2_score(y_test, test_pred[test_ix])
        r2_train += r2_score(y_train, train_pred)
        fold_mean_x.append(np.mean(x_test,axis=0))
        fold_mean_y.append(np.mean(y_test))
    r2_test /= folds
    r2_train /= folds
 

    return test_pred, r2_test, r2_train


# %% 
"""
This function takes a numpy array and returns concatenated data based on start and stop indices
"""
def concat_data_given_start_stop_indices(dataset, start_idx, stop_idx):
    assert len(start_idx) == len(stop_idx), "Start and stop indices lists must be of the same length"
    
    data_slices = []
    for start, stop in zip(start_idx, stop_idx):
        data_slices.append(dataset[start:stop])
    
    concatenated_data = np.concatenate(data_slices, axis=0)
    
    return concatenated_data
# %%
"""
Takes column name and returns predicted kinematic data, true kinematic data, and r2 score
"""

def column_name_to_predict_and_r2(column_name:str, start_idx: List[int], stop_idx: List[int], alpha, folds: int, use_smooth_data: bool = False):


    kin_slice, rates_slice = return_all_nonNan_slice(column_name, use_smooth_data=use_smooth_data)
    vel = diff_filter(kin_slice)
    regression_rates_slice = np.log(rates_slice + 1e-10)

    # slice vel and rates_slice to exclude handpicked_outliers

    regression_vel_slice = concat_data_given_start_stop_indices(vel, start_idx, stop_idx)
    regression_rates_slice = concat_data_given_start_stop_indices(regression_rates_slice, start_idx, stop_idx)

    # # plot rates slice using pcolor
    # fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    # c = ax.pcolor(regression_rates_slice.T, cmap='viridis')
    # ax.set_title('Rates Slice')
    # ax.set_xlabel('Time (bins)')
    # ax.set_ylabel('Channels')
    # fig.colorbar(c, ax=ax, label='Firing Rate')
    # plt.show()


    #determine outliers of velocity   
    outlier_min = np.argmin(regression_vel_slice)
    outlier_max = np.argmax(regression_vel_slice)
    regression_vel_slice = np.delete(regression_vel_slice, [outlier_min, outlier_max])
    regression_rates_slice = np.delete(regression_rates_slice, [outlier_min, outlier_max], axis=0)

    var = np.var(regression_rates_slice, axis=0)
    large_variance_channels = np.where(var > 5)
    regression_rates_slice = np.delete(regression_rates_slice, large_variance_channels, axis=1) # remove channels with large variance

    # remove outliers
    
    # Predict kinematic data from lfads rates
    #predicted_vel, _, r2_sklearn = cross_pred(regression_rates_slice, regression_vel_slice, alpha=alpha, kfolds=10)

    pred_vel, r2_test, r2_train = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha, folds=folds)
    return pred_vel, regression_vel_slice, r2_test, r2_train


# %%
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
fig.suptitle('Kinematic Data')

big_ax = fig.add_subplot(111, frameon=False)
# Hide tick and tick label of the big subplot
big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
big_ax.grid(False)

# Set the labels
big_ax.set_xlabel('Time (bins)', labelpad=5)
big_ax.set_ylabel('Velocity', labelpad=0)


ax = axs.flatten()
#alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100, 1000, 10000]
alpha_values = [1e-10, 1e-6, 1e-2, 1e-1, 1, 3, 10, 100]

best_alpha = None
best_r2 = -np.inf
r2_values_test_all = [] 
r2_values_train_all = []
folds = [10]

for fold in folds:
    r2_test_value_fold = []
    r2_train_value_fold = []
    for i, alpha in enumerate(alpha_values):

        for idx, column_name in enumerate(dataset.data.kin_pos.columns):
            if idx > 0:
                break
            predicted_vel, vel, r2_test, r2_train = column_name_to_predict_and_r2(column_name, start_idx=[9650], stop_idx=[9650+600], alpha=alpha, use_smooth_data=True, folds=fold)
            r2_test_value_fold.append(r2_test)
            r2_train_value_fold.append(r2_train)
            
            ax[i].plot(vel, label='True')
            ax[i].plot(predicted_vel, label='Predicted')
            title = f"r^2 test: {round(r2_test, 3)} r^2 train: {round(r2_train, 3)}, alpha: {alpha}, folds: {fold}"

            ax[i].set_title(title)
            ax[i].set_xlabel('Time (bins)')
            ax[i].set_ylabel('Velocity')
            ax[i].legend()
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    plt.show()
    r2_values_test_all.append(r2_test_value_fold)
    r2_values_train_all.append(r2_train_value_fold)


# %% 
for i in range(len(folds)):
    plt.plot(alpha_values, r2_values_test_all[i], label=f'{folds[i]} folds test')
    plt.plot(alpha_values, r2_values_train_all[i], label=f'{folds[i]} folds train')

plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.title('R^2 vs Alpha')

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R^2')


plt.legend()
plt.show()

# %%
# Toy dataset 
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=600, n_features=1, noise=0.5)
folds = [1,5,10,20]
r2_values_test_all = []
r2_values_train_all = []
alpha_values = [1e-10, 1e-2, 1, 10, 100]
plot = True
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for fold in folds:
    r2_test_value_fold = []
    r2_train_value_fold = []
    for alpha in alpha_values:

        pred, r2_test, r2_train = linear_regression_train_val(X, y, alpha=alpha, folds=fold)
        if plot and fold == 10 and alpha == 1e-2:
            axs[0].plot(X, y, label='True')
            axs[0].plot(X, pred, label='Predicted')
            axs[0].set_title(f'Predicted vs True.')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].legend()

            plot = False
        r2_test_value_fold.append(r2_test)
        r2_train_value_fold.append(r2_train)

    r2_values_test_all.append(r2_test_value_fold)
    r2_values_train_all.append(r2_train_value_fold)

for i in range(len(folds)):
    axs[1].scatter(alpha_values, r2_values_test_all[i], label=f'{folds[i]} folds test')
    axs[1].plot(alpha_values, r2_values_train_all[i], label=f'{folds[i]} folds train')

    
axs[1].set_title('R^2 vs Alpha on linear toy dataset - works as expected')

axs[1].set_xscale('log')
axs[1].set_xlabel('Alpha')
axs[1].set_ylabel('R^2')


axs[1].legend()
plt.show()


# %%
kin_slice, rates_slice = return_all_nonNan_slice('ankle_x', use_smooth_data=False)
vel = diff_filter(kin_slice)
vel = vel[9650:9650+600]
rates_slice = rates_slice[9650:9650+600] 

# plot rates slice using pcolor
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
c = ax[0].pcolor(rates_slice.T, cmap='viridis')
ax[0].set_title('Rates Slice')
ax[0].set_xlabel('Time (bins)')
ax[0].set_ylabel('Channels')
fig.colorbar(c, ax=ax, label='Firing Rate')

# plot vel

ax[1].plot(vel)
ax[1].set_title('Velocity')
ax[1].set_xlabel('Time (bins)')
ax[1].set_ylabel('Velocity')
plt.show()


# %%
