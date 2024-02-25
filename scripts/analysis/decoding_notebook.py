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

import dill
from analysis_utils import *
import scipy.signal as signal

from sklearn.linear_model import Ridge
import typing
from typing import List
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score




# %%
# Load everything

# we use smoothed kinematic data for decoding so we don't have to smooth it every time
load_smooth_kin_data = True

path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)
smooth_kin_data_path = "/snel/share/share/derived/scpu_snel/nwb_lfads/runs/binsize_2/NG_smooth_kin_data.pkl"
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"

if load_smooth_kin_data:
    with open(smooth_kin_data_path,'rb') as inf:
        dataset = pkl.load(inf)
        bin_size = dataset.bin_size

        
else:

    dataset, bin_size = load_dataset_and_binsize(yaml_config_path)
    dataset.smooth_spk(signal_type='kin_pos', gauss_width=3, name='smooth_3', overwrite=False)
    with open(smooth_kin_data_path, "wb") as f:
        dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    # dataset has kinematic data, smoothed spikes, and smoothed lfads_factors

# %% 
#dataset.smooth_spk(signal_type='kin_pos', gauss_width=20, name='smooth_20', overwrite=False)
gauss_width_ms = 10
gauss_width_s = gauss_width_ms/1000
# cutoff frequency using -3db cutoff
cutoff_freq = (2 * np.sqrt(2 * np.log(2))) / (gauss_width_s*(np.sqrt(np.pi)))
print(f'Cutoff frequency: {cutoff_freq} Hz')
#dataset.smooth_spk(signal_type='spikes', gauss_width=100, name='smooth_100', overwrite=False)
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
        all_kin_data = dataset.data.kin_pos_smooth_20
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

    r2_test = np.zeros(folds)
    r2_train = np.zeros(folds)
    fold_mean_x = [] # for test data
    fold_mean_y = [] # for test data
    test_pred = np.zeros_like(y)

    for split_num, (train_ix, test_ix) in enumerate(kf.split(x)):
        x_train, x_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]


        lr.fit(x_train, y_train)
        test_pred[test_ix] = lr.predict(x_test)
        train_pred = lr.predict(x_train)
        r2_test[split_num] = r2_score(y_test, test_pred[test_ix])
        r2_train[split_num] = r2_score(y_train, train_pred)
        
        fold_mean_x.append(np.mean(x_test,axis=0))
        fold_mean_y.append(np.mean(y_test))
    
    test_sem = np.std(r2_test) / np.sqrt(folds)
    train_sem = np.std(r2_train) / np.sqrt(folds)
 
    r2_test = np.mean(r2_test)
    r2_train = np.mean(r2_train)


    return test_pred, r2_test, r2_train, train_sem, test_sem


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

def plot_rates_vel_reg(regression_vel_slice, regression_rates_slice):
    """
    Plots velocity and rates (input to linear regression model)
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(regression_vel_slice)
    ax[0].set_title('Velocity')
    ax[0].set_xlabel('Time (bins)')
    ax[0].set_ylabel('Velocity')
    vmax = np.max(regression_rates_slice)
    print(vmax)
    c = ax[1].pcolor(regression_rates_slice.T, cmap='viridis', vmin=0, vmax=vmax)
    ax[1].set_title('Rates')
    ax[1].set_xlabel('Time (bins)')
    ax[1].set_ylabel('Rates')
    plt.colorbar(c)
    plt.show()

def plot_psd_kinematic_data(kin_slice, use_smooth_data, bin_size=2):
    """
    Plots power spectral density of kinematic data
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fs = 1/(bin_size/1000) # sampling frequency in Hz
    f, Pxx_den = signal.welch(kin_slice, fs, nperseg=1024)
    ax.semilogy(f, Pxx_den)
    title = f'PSD: kinematic data Smooth: cutoff: {cutoff_freq}' if use_smooth_data else f'PSD: kinematic data'
    ax.set_title(title)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    plt.show()

def preprocessing(column_name, use_smooth_data, start_idx, stop_idx, plot=False):
    """
        preprocessing steps. output will be used for linear regression
    """
    # PREPROCESSING
    kin_slice, rates_slice = return_all_nonNan_slice(column_name, use_smooth_data=use_smooth_data)
    regression_vel_slice = diff_filter(kin_slice)
    # regression_rates_slice = np.log(rates_slice + 1e-10)
    regression_rates_slice = rates_slice


    if len(start_idx) >= 1 and len(stop_idx) >= 1 : # if start and stop indices are provided
        regression_vel_slice = concat_data_given_start_stop_indices(regression_vel_slice, start_idx, stop_idx)
        regression_rates_slice = concat_data_given_start_stop_indices(regression_rates_slice, start_idx, stop_idx)


    index_smallest_10 = np.argpartition(regression_vel_slice, 10)[:10]
    index_largest_10 = np.argpartition(regression_vel_slice, -10)[-10:]
    outlier_indices = np.concatenate((index_smallest_10, index_largest_10))
    
    # outlier_min = np.argmin(regression_vel_slice)
    # val_min = regression_vel_slice[outlier_min]
    # print(val_min)
    # outlier_max = np.argmax(regression_vel_slice)
    regression_vel_slice = np.delete(regression_vel_slice, outlier_indices)
    regression_rates_slice = np.delete(regression_rates_slice, outlier_indices, axis=0)

    # apply wiener filter
    # m = 20
    # regression_vel_slice = np.convolve(regression_vel_slice, np.ones((m,))/m, mode='same')

    # delete channels with large variance
    # var = np.var(regression_rates_slice, axis=0)
    # large_variance_channels = np.where(var > 5)
    # regression_rates_slice = np.delete(regression_rates_slice, large_variance_channels, axis=1) # remove channels with large variance
    if plot:
        plot_rates_vel_reg(regression_vel_slice, regression_rates_slice)
        plot_psd_kinematic_data(regression_vel_slice, use_smooth_data, bin_size=2)
    return regression_vel_slice, regression_rates_slice


# %%
use_smooth_data = False

# fig, axs = plt.subplots(3, 2, figsize=(10, 12))
# fig.suptitle(f'Kinematic Data vs Predicted Kinematic Data for different alphas and folds. Smoothed: {use_smooth_data}')

# big_ax = fig.add_subplot(111, frameon=False)
# # Hide tick and tick label of the big subplot
# big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# big_ax.grid(False)

# # Set the labels
# big_ax.set_xlabel('Time (bins)', labelpad=5)
# big_ax.set_ylabel('Velocity', labelpad=0)


# ax = axs.flatten()
#alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100, 1000, 10000]
alpha_values = np.logspace(-7, 3, num=7)

best_alpha = None
best_r2 = -np.inf
r2_values_test_all = [] 
r2_values_train_all = []
train_sem_all = [] 
test_sem_all = [] 
fold = 10
r2_test_value_fold = []
r2_train_value_fold = []

for idx, column_name in enumerate(dataset.data.kin_pos.columns):
    if idx > 0:
        # only decode one column for now
        break
    regression_vel_slice, regression_rates_slice = preprocessing(column_name, use_smooth_data, start_idx=[9650], stop_idx=[9650+600], plot=True)

    for i, alpha in enumerate(alpha_values):
        print(f'Alpha: {alpha}')

        # 9650 start index, 9650+600 stop index was best for 
        predicted_vel, r2_test, r2_train, train_sem, test_sem = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha, folds=fold)
        r2_test_value_fold.append(r2_test)
        r2_train_value_fold.append(r2_train)
        train_sem_all.append(train_sem)
        test_sem_all.append(test_sem)
        
        # ax[i].plot(vel, label='True')
        # ax[i].plot(predicted_vel, label='Predicted (Test)')
        # title = f"r^2 test: {round(r2_test, 3)} r^2 train: {round(r2_train, 3)}, alpha: {alpha}, folds: {fold}"

        # ax[i].set_title(title)
        # ax[i].set_xlabel('Time (bins)')
        # ax[i].set_ylabel('Velocity')
        # ax[i].legend()
        # ax[i].spines['right'].set_visible(False)
        # ax[i].spines['top'].set_visible(False)

        if r2_test > best_r2:
            best_r2 = r2_test
            best_alpha = alpha
            best_pred = predicted_vel
            best_true = regression_vel_slice
            best_test_sem = test_sem
            best_train_sem = train_sem

    # fig.subplots_adjust(hspace=0.5)
    # fig.subplots_adjust(wspace=0.5)
    # plt.show()


# %%
    
# plot best alpha
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("LFADS Rates")
ax[0].plot(best_true, label='True')
ax[0].plot(best_pred, label='Predicted')
ax[0].set_title(f'Predicted vs True. Best Alpha: {round(best_alpha, 3)} $R^2$: {round(best_r2, 3)}')
ax[0].set_xlabel('Time (bins)')
ax[0].set_ylabel('Velocity')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

ax[0].legend()



r2_test_value_fold = np.array(r2_test_value_fold)
r2_train_value_fold = np.array(r2_train_value_fold)
train_sem_all = np.array(train_sem_all)
test_sem_all = np.array(test_sem_all)


ax[1].plot(alpha_values, r2_test_value_fold, '-o', label=f'10 folds validation')
ax[1].fill_between(alpha_values, r2_test_value_fold - train_sem_all, r2_test_value_fold + train_sem_all, alpha=0.5)

ax[1].plot(alpha_values, r2_train_value_fold, '-o',label=f'10 folds train')
ax[1].fill_between(alpha_values, r2_train_value_fold - test_sem_all, r2_train_value_fold + test_sem_all, alpha=0.1)

ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('$R^2$')
ax[1].set_title('$R^2$ vs Alpha')

ax[1].set_xscale('log')



ax[1].legend()
plt.show()

# %%
column_name = 'ankle_x'
kin_slice, rates_slice = return_all_nonNan_slice(column_name, use_smooth_data=use_smooth_data)
regression_vel_slice = diff_filter(kin_slice)
regression_rates_slice = np.log(rates_slice + 1e-10)

smallest_5 = np.partition(regression_vel_slice, 10)[:10]
index_smallest_5 = np.argpartition(regression_vel_slice, 5)[:10]
print(smallest_5)
print(index_smallest_5)
# %%
