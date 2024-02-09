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
import matplotlib.cm as cm
import matplotlib.colors as colors




# %%
# Load everything
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)

dataset, bin_size = load_dataset_and_binsize(yaml_config_path)
# dataset has kinematic data, smoothed spikes, and smoothed lfads_factors

# %% 
# Smooth kinematic data
#dataset.smooth_spk(signal_type='kin_pos', gauss_width=8, name='smooth_8', overwrite=False)
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

# rates = dataset.data.lfads_rates.values
# for i in range(4):
#     fig, axs = plt.subplots(1, 1, figsize=(20, 12))
#     axs.plot(rates[:, i])
#     axs.set_title(f'Channel {i}')
#     axs.set_xlabel('Bins')
#     axs.set_ylabel('Firing rate')
#     plt.show()

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

def return_upper_outliers(data: np.ndarray) -> np.ndarray:
    """
    Returns indices of upper outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_bound = q3 + 6 * iqr
    indices = np.where(data > upper_bound)[0]
    return indices
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
        """
        variance channels sorted descending order
        [5.16522246 2.93424483 2.57730398 2.5385062  2.40587629 2.32626326
 2.23327391 2.21452203 2.13587281 2.10802132 2.0926351  2.06493209
 2.05003696 2.03460004 2.00813071 1.96947407 1.95340772 1.88638498
 1.84227467 1.67509189 1.67095371 1.64857727 1.6115832  1.60164427
 1.57764281 1.54246901 1.51767094 1.47130332 1.47068484 1.46039802
 1.38324884 1.34198107 1.28362665 1.23286746 1.20691066 1.18028282
 1.10306594 1.02229998 0.99460861 0.99460616 0.96967062 0.96166463
 0.94271015 0.94121756 0.8995361  0.89527348 0.88847127 0.85299992
 0.80059576 0.79441946 0.76340501 0.76146608 0.70342064 0.69882349
 0.69864316 0.69397564 0.67870717 0.66734888 0.60803599 0.60129593
 0.59056843 0.57843648 0.57570408 0.55761227 0.54901659 0.5274489
 0.51840433 0.49899345 0.48593633 0.45806388 0.41386854 0.40334881
 0.38496376 0.37464683 0.37035802 0.35268387 0.31074842 0.29885579
 0.27227127 0.23743894 0.23719647 0.23056927 0.13322168 0.10682749
 0.08867252 0.07373564 0.04054569 0.03784658 0.01746072]

        [28 16 81 33  5 86 88 82 24 42 20 30 75 39 77 37 19 29 67 80 18 73 72 62
  3 84 70 31  6 47 66 36 43 61 23 53  0 34 50 76 15 17 59 74 14 11  8 51
 22 64 12 78 27 83 55 71 60 63 32 56 35 65 69  1 21 45 13 38  7 52 44 10
 79 57  2 41 68 58 48  9 87 46  4 85 40 25 49 54 26]"""
        kf = KFold(n_splits=folds)
        lr = Ridge(alpha=alpha)
        test_pred = np.zeros_like(y)
        train_pred = np.zeros_like(y)
        r2_test = 0
        r2_train = 0
        fold_mean_x = [] # for test data
        fold_mean_y = [] # for test data
        large_variance_channels = [28,61,87,33,5,86,88,82,24,42,20, 30]
        x = np.delete(x, large_variance_channels, axis=1)
        print(x.shape)
        for train_ix, test_ix in kf.split(x):
            x_train, x_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            lr.fit(x_train, y_train)
            test_pred[test_ix] = lr.predict(x_test)
            train_pred[train_ix] = lr.predict(x_train)
            r2_test += r2_score(y_test, test_pred[test_ix])
            fold_mean_x.append(np.mean(x_test,axis=0))
            fold_mean_y.append(np.mean(y_test))
        r2_test /= folds
        lr.fit(x, y)
        train_pred = lr.predict(x)
        r2_train = r2_score(y, train_pred)
        # get 0th channel for each fold
        fold_mean_x = np.array(fold_mean_x) # (fold, channel) each element is mean of channel
        fold_mean_y = np.array(fold_mean_y) # (fold,) each element is mean of kinematic data
        fold_mean_x /= np.mean(x, axis=0) # normalize mean
        variances = np.var(fold_mean_x, axis=0) # variance of each channel

        indices = np.argsort(variances)[::-1] # descending order

        fold_mean_x = fold_mean_x[:, indices] # sort channels by variance
        sorted_variances = variances[indices]
        print(sorted_variances)
        print(indices)
        # # Normalize variances to [0,1] for color mapping
        # normalized_variances = variances[indices] / np.max(variances)

        # # Create a colormap
        # cmap = cm.get_cmap('viridis')
        # norm = colors.Normalize(vmin=np.min(variances), vmax=np.max(variances))

        # subplot
        # fig, axs = plt.subplots(1, 2, figsize=(10, 12))
        # axs[0].plot(range(folds), fold_mean_y, label='Fold Mean Rates')
        # axs[0].set_xlabel('Fold')
        # axs[0].set_ylabel('Mean')
        # axs[0].set_title('Mean of Predicted Values for Each Fold')
        # axs[0].legend()
        # for chan_num in range(fold_mean_x.shape[1]):
        #     plt.plot(range(folds), fold_mean_x[:, chan_num], 'o-', color=cmap(normalized_variances[chan_num]), label=f'Channel {indices[chan_num]}')

        # plt.xlabel('Fold')
        # plt.ylabel('Mean Normalized')
        # plt.title(f'Mean of Rates Data for every channel. # of Fold {folds} and alpha {alpha}')
        # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Variance')

        # plt.show()

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

    # determine outliers of velocity   
    outlier_min = np.argmin(regression_vel_slice)
    outlier_max = np.argmax(regression_vel_slice)
    regression_vel_slice = np.delete(regression_vel_slice, [outlier_min, outlier_max])
    regression_rates_slice = np.delete(regression_rates_slice, [outlier_min, outlier_max], axis=0)





    # remove outliers
    
    # Predict kinematic data from lfads rates
    #predicted_vel, _, r2_sklearn = cross_pred(regression_rates_slice, regression_vel_slice, alpha=alpha, kfolds=10)

    predicted_vel, r2_test, r2_train = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha, folds=folds)
    return predicted_vel, regression_vel_slice, r2_test, r2_train


# %%
# fig, axs = plt.subplots(1, 1, figsize=(10, 12))
# fig.suptitle('Kinematic Data')

# big_ax = fig.add_subplot(111, frameon=False)
# # Hide tick and tick label of the big subplot
# big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# big_ax.grid(False)

# # Set the labels
# big_ax.set_xlabel('Time (bins)', labelpad=5)
# big_ax.set_ylabel('Velocity', labelpad=0)


# ax = axs.flatten()
#alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100, 1000, 10000]
alpha_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
best_alpha = None
best_r2 = -np.inf
r2_values_test_all = [] 
r2_values_train_all = []
folds = [10]

for fold in folds:
    r2_test_value_fold = []
    r2_train_value_fold = []
    for alpha in alpha_values:

        for idx, column_name in enumerate(dataset.data.kin_pos.columns):
            if idx > 0:
                break
            predicted_vel, vel, r2_test, r2_train = column_name_to_predict_and_r2(column_name, start_idx=9500+150, stop_idx=9500+750, alpha=alpha, use_smooth_data=False, folds=fold)
            r2_test_value_fold.append(r2_test)
            r2_train_value_fold.append(r2_train)
            
            # axs.plot(vel, label='True')
            # axs.plot(predicted_vel, label='Predicted')
            # title = f"r^2 test: {r2_test} r^2 train: {r2_train}, alpha: {alpha}, folds: {fold}"

            # axs.set_title(title)
            # axs.set_xlabel('Time (bins)')
            # axs.set_ylabel('Velocity')
            # axs.legend()
            # axs.spines['right'].set_visible(False)
            # axs.spines['top'].set_visible(False)

        # fig.subplots_adjust(hspace=0.5)
        # plt.show()
    r2_values_test_all.append(r2_test_value_fold)
    r2_values_train_all.append(r2_train_value_fold)

# %% 
for i in range(len(folds)):
    plt.plot(alpha_values, r2_values_test_all[i], label=f'{folds[i]} folds test')
    plt.plot(alpha_values, r2_values_train_all[i], label=f'{folds[i]} folds train')

plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.title('R^2 vs Alpha removed channel 28, 16, 81, 33, 5, 86, 88, 82, 24, 42, 20, 30')

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
