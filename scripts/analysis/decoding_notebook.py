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
# plot factors over time
# factors = dataset.data.lfads_factors.values

# fig, axs = plt.subplots(1, 1, figsize=(20, 12))
# axs.plot(factors[:, 0])
# axs.set_title('Factor 1')
# axs.set_xlabel('Bins')
# axs.set_ylabel('Arbitrary Units')
# plt.show()

# plot rates over time

rates = dataset.data.lfads_rates.values
for i in range(4):
    fig, axs = plt.subplots(1, 1, figsize=(20, 12))
    axs.plot(rates[:, i])
    axs.set_title(f'Channel {i}')
    axs.set_xlabel('Bins')
    axs.set_ylabel('Firing rate')
    plt.show()

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


def linear_regression_train_val(x, y, alpha=0, folds=5):
    """
    x is 2D array of shape (n_bins, n_channels) which is (samples, features)
    y is 1D array of shape (n_bins,) where each value is kinematic data
    """

    if folds == 1:
        lr = Ridge(alpha=alpha)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
        lr.fit(x_train, y_train)
        test_pred = lr.predict(x_test)
        train_pred = lr.predict(x_train)

        r2_test = r2_score(y_test, test_pred)
        r2_train = r2_score(y_train, train_pred)
    else:
        kf = KFold(n_splits=folds, shuffle=True)
        lr = Ridge(alpha=alpha)
        test_pred = np.zeros_like(y)
        train_pred = np.zeros_like(y)
        r2_test = 0
        r2_train = 0
        for train_ix, test_ix in kf.split(x):
            x_train, x_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            lr.fit(x_train, y_train)
            test_pred[test_ix] = lr.predict(x_test)
            train_pred[train_ix] = lr.predict(x_train)
            r2_test += r2_score(y_test, test_pred[test_ix])
        r2_test /= folds
        lr.fit(x, y)
        train_pred = lr.predict(x)
        r2_train = r2_score(y, train_pred)

        
        plt.plot(y, label='True')
        plt.plot(test_pred, label='Predicted')
        plt.title(f"Linear Regression folds={folds}, alpha={alpha}\nR^2 test = {r2_test}, R^2 train = {r2_train}")
        plt.legend()
        plt.show()

    return test_pred, r2_test, r2_train

# %%
"""
Takes column name and returns predicted kinematic data, true kinematic data, and r2 score
"""

def column_name_to_predict_and_r2(column_name:str, start_idx: int, stop_idx: int, alpha, folds: int, use_smooth_data: bool = False):


    kin_slice, rates_slice = return_all_nonNan_slice(column_name, use_smooth_data=use_smooth_data)
    vel = diff_filter(kin_slice)
    #rates_slice = np.log(rates_slice + 1e-6)

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

    predicted_vel, r2_test, r2_train = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha, folds=folds)
    return predicted_vel, regression_vel_slice, r2_test, r2_train

fig, axs = plt.subplots(1, 1, figsize=(10, 12))
fig.suptitle('Kinematic Data')

big_ax = fig.add_subplot(111, frameon=False)
# Hide tick and tick label of the big subplot
big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
big_ax.grid(False)

# Set the labels
big_ax.set_xlabel('Time (bins)', labelpad=5)
big_ax.set_ylabel('Velocity', labelpad=0)


# ax = axs.flatten()
alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100, 1000, 10000]

best_alpha = None
best_r2 = -np.inf
r2_values_test_all = [] 
r2_values_train_all = []
folds = [3]

for fold in folds:
    r2_test_value_fold = []
    r2_train_value_fold = []
    for alpha in alpha_values:

        for idx, column_name in enumerate(dataset.data.kin_pos.columns):
            if idx > 0:
                break
            predicted_vel, vel, r2_test, r2_train = column_name_to_predict_and_r2(column_name, start_idx=9500+150, stop_idx=9500+750, alpha=alpha, use_smooth_data=True, folds=fold)
            r2_test_value_fold.append(r2_test)
            r2_train_value_fold.append(r2_train)
            
            axs.plot(vel, label='True')
            axs.plot(predicted_vel, label='Predicted')
            title = f"r^2 test: {r2_test} r^2 train: {r2_train}, alpha: {alpha}, folds: {fold}"

            axs.set_title(title)
            axs.set_xlabel('Time (bins)')
            axs.set_ylabel('Velocity')
            axs.legend()
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)

        fig.subplots_adjust(hspace=0.5)
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
plt.plot(alpha_values, r2_values_test_all[0], label=f'{folds[0]} test')
plt.plot(alpha_values, r2_values_train_all[0], label=f'{folds[0]} train')
plt.title("R^2 vs Alpha for 1 fold")
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
# %%
plt.plot(alpha_values, r2_values_test_all[1], label=f'{folds[1]} test')
plt.plot(alpha_values, r2_values_train_all[1], label=f'{folds[1]} train')
plt.title("R^2 vs Alpha for 2 fold")
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
# %%
np.random.seed(0)  # For reproducibility
n_samples = 100 # equivalent to number of bins
n_features = 4 # equivalent to number of channels

# Generate random data for features
X = np.random.normal(size=(n_samples, n_features))

# Generate target values as a linear function of the features
# y = 3*X + 4 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + 4 * X[:, 3] + np.random.normal(size=n_samples)
folds = 3
test_pred, r2_test, r2_train = linear_regression_train_val(X, y, alpha=1e-8, folds=folds)
r2_test = round(r2_test, 4)
r2_train = round(r2_train, 4)
alpha = 1e-8
plt.plot(y, label='True')
plt.plot(test_pred, label='Predicted')
plt.xlabel('Time (au)')
plt.ylabel('Amplitude (au)')
plt.title(f'Linear Regression with folds = {folds} and alpha = {alpha}\nR^2 test = {r2_test} , R^2 train = {r2_train}')
plt.legend()
plt.show()
# %%
