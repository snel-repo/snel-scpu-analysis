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
gauss_width_ms = 100
gauss_width_s = gauss_width_ms/1000
# cutoff frequency using -3db cutoff
cutoff_freq = (2 * np.sqrt(2 * np.log(2))) / (gauss_width_s*(np.sqrt(np.pi)))
print(f'Cutoff frequency: {cutoff_freq} Hz')
# %%
#dataset.smooth_spk(signal_type='lfads_rates', gauss_width=gauss_width_ms, name=f'smooth_{gauss_width_ms}', overwrite=False)
dataset.smooth_spk(signal_type='spikes', gauss_width=gauss_width_ms, name='smooth_25', overwrite=False)
# dataset.smooth_spk(signal_type='lfads_factors', gauss_width=8, name='smooth_8', overwrite=False)
# dataset.data.lfads_factors_smooth_8 = dataset.data.lfads_factors_smooth_8.fillna(0)

# %% 

# print out the ms of the trial start and ends
for index, trial in dataset.trial_info.iterrows():
    if trial['event_type'] == 'locomotion':
        trial_id = trial['trial_id']
        start_time_ms = trial['start_time'].total_seconds() * 1000
        end_time_ms = trial['end_time'].total_seconds() * 1000
        # convert to bins of size dataset.bin_size
        start_time_bin = int(start_time_ms / bin_size)
        end_time_bin = int(end_time_ms / bin_size)
        print(f"Start bin: {start_time_bin}, End bin: {end_time_bin} for trial {trial_id}")

# %%
'''
This function takes a dataframe column and returns indices where data is not NaN

Parameters:
    dataframe_column: pandas dataframe column

Returns:
    trials: list of tuples where each tuple is a start and end index of where data is not NaN
'''
def get_change_indices(dataframe_column: pd.Series) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Find NaNs
    is_nan = dataframe_column.isna()

    # Find Changes in State
    start_indices = np.where(~is_nan & is_nan.shift(1, fill_value=True))[0]
    end_indices = np.where(is_nan & ~is_nan.shift(1, fill_value=True))[0] - 1
    assert len(start_indices) == len(end_indices)
    print(f"start indices: {start_indices}\nend indices: {end_indices}")
    return start_indices, end_indices

"""
This function takes a numpy array and returns concatenated data based on start and stop indices
"""
def concat_data_given_start_stop_indices(dataset: np.ndarray, start_indices: List[int], stop_indices: List[int]) -> np.ndarray:
    assert len(start_indices) == len(stop_indices), "Start and stop indices lists must be of the same length"
    
    concatenated_data = np.concatenate(tuple(dataset[start:stop] for start, stop in zip(start_indices, stop_indices)), axis=0)
    orig_indices = orig_indices_to_new_indices(start_indices, stop_indices)
    
    return concatenated_data, orig_indices
# %%
def orig_indices_to_new_indices(start_indices: List[int], end_indices: List[int]) -> List[int]:
    """
        Given the start and end indices of non-NaN data, make a list of indices that correspond to the original indices
        For example: start_indices = [0, 10, 100] and end_indices = [5, 15, 105], then the function will return [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 100, 101, 102, 103, 104, 105]
    """
    orig_indices = []
    for start_index, stop_index in zip(start_indices, end_indices):
        orig_indices.extend(range(start_index, stop_index+1))
    return orig_indices

# %%

"""
Returns all nonNan data of kinematic data for all given body parts and returns lfads rates based on those indices as well as the column names

"""
def return_all_nonNan_slice(use_smooth_data: bool = True, use_LFADS: bool = True) -> typing.Tuple[pd.DataFrame, np.ndarray, List[str]]:

    
    
    if use_smooth_data:
        all_kin_data = dataset.data.kin_pos_smooth_3
        if use_LFADS:
            all_rates_data = dataset.data.lfads_rates_smooth_8
        else:
            all_rates_data = dataset.data.spikes_smooth_25 # smoothing LFADS rates makes R^2 improve the most compared to smoothing kinematic data
        #all_rates_data = dataset.data.lfads_rates_smooth_50
    else:
        all_kin_data = dataset.data.kin_pos
        if use_LFADS:
            all_rates_data = dataset.data.lfads_rates
        else:
            all_rates_data = dataset.data.spikes_smooth_25

        #all_rates_data = dataset.data.spikes_smooth_100
    kin_column_names = all_kin_data.columns
    start_indices, end_indices = get_change_indices(all_kin_data["hip_x"]) # must use hip_x/hip_y to get change indices for all body parts because hip_x/hip_y is missing some data where there is data for other body parts
    
    # orig_indices is a list of the original indices that correspond to the new indices
    all_valid_kin_data, orig_indices = concat_data_given_start_stop_indices(all_kin_data.values, start_indices, end_indices) # .values to convert to numpy array because below function expects numpy array

    rates_slice, _ = concat_data_given_start_stop_indices(all_rates_data.values, start_indices, end_indices)
    # convert kinematic data back to pandas dataframe 
    return pd.DataFrame(all_valid_kin_data, columns=kin_column_names), rates_slice, kin_column_names, orig_indices


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

    kf = KFold(n_splits=folds,  shuffle=True, random_state=42)
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

def append_kinematic_angle_data(all_kin_df: pd.DataFrame) -> pd.DataFrame:
    """
        Calculates and Appends knee and ankle angle data to kinematic data and returns all kin data and angles in one dataframe
    """
    # get knee and ankle angle data
    ankle_kinematics_np = all_kin_df[["ankle_x", "ankle_y"]].values
    hip_kinematics_np = all_kin_df[["hip_x", "hip_y"]].values
    iliac_crest_kinematics_np = all_kin_df[["iliac_crest_x", "iliac_crest_y"]].values
    knee_kinematics_np = all_kin_df[["knee_x", "knee_y"]].values
    toe_kinematics_np = all_kin_df[["toe_x", "toe_y"]].values

    hip_to_knee_vectors = knee_kinematics_np - hip_kinematics_np
    knee_to_ankle_vectors = ankle_kinematics_np - knee_kinematics_np
    ankle_to_toe_vectors = toe_kinematics_np - ankle_kinematics_np

    # Dot product and magnitudes for knee angle
    dot_product_knee = np.sum(hip_to_knee_vectors * knee_to_ankle_vectors, axis=1)
    mag_hip_to_knee = np.linalg.norm(hip_to_knee_vectors, axis=1)
    mag_knee_to_ankle = np.linalg.norm(knee_to_ankle_vectors, axis=1)

    # Dot product and magnitudes for ankle angle
    dot_product_ankle = np.sum(knee_to_ankle_vectors * ankle_to_toe_vectors, axis=1)
    mag_ankle_to_toe = np.linalg.norm(ankle_to_toe_vectors, axis=1)

    # Angle calculation
    knee_angles_vectorized = np.arccos(dot_product_knee / (mag_hip_to_knee * mag_knee_to_ankle)) * (180 / np.pi)
    ankle_angles_vectorized = np.arccos(dot_product_ankle / (mag_knee_to_ankle * mag_ankle_to_toe)) * (180 / np.pi)

    
    all_kin_df["knee_angle"] = knee_angles_vectorized
    all_kin_df["ankle_angle"] = ankle_angles_vectorized

    return all_kin_df


# %% 


def plot_rates_vel_reg(regression_vel_slice, regression_rates_slice, start_idx, stop_idx):
    """
    Plots velocity and rates (input to linear regression model)
    """
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
    fig.suptitle(f'Velocity and Rates indices {start_idx, stop_idx}')
    fig.subplots_adjust(hspace=0.25)

    ax[0].plot(regression_vel_slice)
    ax[0].set_title('Velocity')
    ax[0].set_xlabel('Time (bins)')
    ax[0].set_ylabel('Velocity')
    vmax = np.max(regression_rates_slice)
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
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fs = 1/(bin_size/1000) # sampling frequency in Hz
    f, Pxx_den = signal.welch(kin_slice, fs, nperseg=256)
    ax.semilogy(f, Pxx_den)
    title = f'PSD: kinematic data Smooth: cutoff: {cutoff_freq}' if use_smooth_data else f'PSD: kinematic data'
    ax.set_title(title)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    plt.show()

def preprocessing(column_name, use_smooth_data, start_idx, stop_idx, plot=False, use_LFADS=True):
    """
        preprocessing steps. output will be used for linear regression
    """
    # PREPROCESSING
    all_kin_df, rates_slice, _, orig_indices = return_all_nonNan_slice(use_smooth_data=use_smooth_data, use_LFADS=use_LFADS)
   
    all_kin_slice_and_angle_df = append_kinematic_angle_data(all_kin_df)

    kin_slice = all_kin_slice_and_angle_df[column_name].values
    regression_vel_slice = kin_slice
    #regression_vel_slice = diff_filter(kin_slice)
    regression_rates_slice = np.log(rates_slice + 1e-10)

    # subselect data from non NaN indices
    if len(start_idx) >= 1 and len(stop_idx) >= 1 : # if start and stop indices are provided

        # subset indices are the indices of nonNan data, so we can use it to get the original indices

        regression_vel_slice, subset_indices = concat_data_given_start_stop_indices(regression_vel_slice, start_idx, stop_idx)
        orig_indices = [orig_indices[i] for i in subset_indices]
        regression_rates_slice, _ = concat_data_given_start_stop_indices(regression_rates_slice, start_idx, stop_idx)
        
        print(f"Original indices: {orig_indices}")


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
        plot_rates_slice = np.exp(regression_rates_slice) - 1e-10
        plot_rates_vel_reg(regression_vel_slice, plot_rates_slice, start_idx, stop_idx)
        #plot_psd_kinematic_data(regression_vel_slice, use_smooth_data, bin_size=2)
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
alpha_values = np.logspace(-5, 2, num=10)


r2_values_test_all = [] 
r2_values_train_all = []
train_sem_all = [] 
test_sem_all = [] 
fold = 10
r2_test_value_fold = []
r2_train_value_fold = []
# these are handpicked indices from the nonNan data. currently, they are based on the first video
start_idx = [0]
stop_idx = [5000] # end times for video in trial_info is wrong so can just choose some arbitrary end time
# start_idx = [9650]
# stop_idx = [9650+600]
# by itself, 9650 start index, 9650+600 stop index  R^2 = 0.748
# by itself, 11000 start index, 11000+400 stop index R^2 = 0.871
# together, R^2 = 0.578

for idx, column_name in enumerate(dataset.data.kin_pos.columns):
    if idx > 0:
        # only decode one column for now
        break
    column_name = "ankle_x"
    regression_vel_slice, regression_rates_slice = preprocessing(column_name, use_smooth_data, start_idx=start_idx, stop_idx=stop_idx, plot=True, use_LFADS=True)
    best_alpha = None
    best_r2 = -np.inf
    for i, alpha in enumerate(alpha_values):

        
        predicted_vel, r2_test, r2_train, train_sem, test_sem = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha, folds=fold)
        print(r2_test)
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



# plot best alpha
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"LFADS Rates indices: {start_idx, stop_idx}")
ax[0].plot(best_true, label='True')
ax[0].plot(best_pred, label='Predicted')
ax[0].set_title(f'Predicted vs True. Best Alpha: {round(best_alpha, 3)} $R^2$: {round(best_r2, 6)}')
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
# block to visualize how R^2 between  as more samples are included
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
use_smooth_data = True
start_idx = [22000]
stop_indices = [[22000+400], [22000+600], [22000+800], [22000+1000], [22000+1200], [22000+1400], [22000+1600], [22000+1800], [22000+2000]]
mean_kin_for_stop_indices = []
var_kin_for_stop_indices = []
num_batches = len(stop_indices)
num_channels = regression_rates_slice.shape[1]
mean_rates_for_stop_indices = np.zeros((num_batches, num_channels)) # mean of rates for each channel as more samples are included
var_rates_for_stop_indices = np.zeros((num_batches, num_channels))  # variance of rates for each channel as more samples are included
dataset_sizes = [stop_idx[0] - start_idx[0] for stop_idx in stop_indices]
alpha_values = np.logspace(-5, 2, num=10)
fold = 10
column_name = 'ankle_x'



r2_across_dataset_sizes_smooth = []
r2_across_dataset_sizes_LFADS = []

for use_lfads_param in [True, False]:
    for i in range(len(stop_indices)):

        best_r2 = -np.inf
        best_alpha = None

        stop_idx = stop_indices[i]
        sample_size = dataset_sizes[i]
        
        regression_vel_slice, regression_rates_slice = preprocessing(column_name, use_smooth_data=use_smooth_data, start_idx=start_idx, stop_idx=stop_idx, plot=False, use_LFADS=use_lfads_param)

        for idx, alpha_x in enumerate(alpha_values):

            predicted_vel, r2_test, r2_train, train_sem, test_sem = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha_x, folds=fold)

            # mean_kin_for_stop_indices.append(np.mean(regression_vel_slice))
            # var_kin_for_stop_indices.append(np.var(regression_vel_slice))

            # # rates distribution statistics

            # mean_rates_for_stop_indices[i, :] = np.mean(regression_rates_slice, axis=0)
            # var_rates_for_stop_indices[i, :] = np.var(regression_rates_slice, axis=0)
            if r2_test > best_r2:
                best_r2 = r2_test
                best_alpha = alpha
        if use_lfads_param:
            r2_across_dataset_sizes_LFADS.append(best_r2)
        else:
            r2_across_dataset_sizes_smooth.append(best_r2)

assert(len(r2_across_dataset_sizes_smooth) == len(stop_indices))
assert(len(r2_across_dataset_sizes_LFADS) == len(stop_indices))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f'Change in R^2 as more samples are included')
axs[0].plot(dataset_sizes, r2_across_dataset_sizes_smooth, '-o', label='Smoothed R^2')
axs[0].set_title('Smoothed R^2')
axs[0].set_xlabel('Sample Size')
axs[0].set_ylabel('R^2')
axs[0].legend()
axs[1].plot(dataset_sizes, r2_across_dataset_sizes_LFADS, '-o', label='LFADS R^2')
axs[1].set_title('LFADS R^2')
axs[1].set_xlabel('Sample Size')
axs[1].set_ylabel('R^2')
axs[1].legend()
plt.show()



# # Flatten the sample sizes and channels to use in scatter plot
# x_indices = np.repeat(np.arange(num_batches), num_channels)
# y_indices = np.tile(np.arange(num_channels), num_batches)

# # Flatten the data array for color coding
# colors = mean_rates_for_stop_indices.flatten()
# # Setting labels and title
# xticks_array = np.array(dataset_sizes).flatten()
# rates_fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# rates_fig.suptitle(f'Statistics of rates for different stop indices')
# ax[0].set_title('Mean Rates')
# ax[0].set_xlabel('Sample Size')
# ax[0].set_ylabel('Channels')
# ax[0].set_xticks(ticks=np.linspace(0, num_batches-1, len(xticks_array)))
# ax[1].set_title('Variance of Rates')
# ax[1].set_xlabel('Sample Size')
# ax[1].set_ylabel('Channels')
# ax[1].set_xticks(ticks=np.linspace(0, num_batches-1, len(xticks_array)))
# ax[0].scatter(x_indices, y_indices, c=mean_rates_for_stop_indices.flatten(), cmap='viridis')
# ax[1].scatter(x_indices, y_indices, c=var_rates_for_stop_indices.flatten(), cmap='viridis')
# rates_fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[0], label='Mean Rates') # normalized color bar
# rates_fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[1], label='Variance of Rates')
# plt.show()

# %% 

for channel in range(num_channels):
    plt.plot(range(num_batches), mean_rates_for_stop_indices[:, channel], label=f'Channel {channel}')

# Add labels and title
plt.xticks(ticks=np.linspace(0, num_batches-1, len(xticks_array)), labels=xticks_array)

plt.xlabel('Batch Number')
plt.ylabel('Mean of Rates')
plt.title('Mean of Rates for Each Channel Over Batches')

plt.show()

for channel in range(num_channels):
    plt.plot(range(num_batches), var_rates_for_stop_indices[:, channel], label=f'Channel {channel}')

# Add labels and title
plt.xticks(ticks=np.linspace(0, num_batches-1, len(xticks_array)), labels=xticks_array)

plt.xlabel('Batch Number')
plt.ylabel('Variance of Rates')
plt.title('Variance of Rates for Each Channel Over Batches')

plt.show()


# %%
plt.scatter(sample_sizes, mean_kin_for_stop_indices, label="mean")
plt.scatter(sample_sizes, var_kin_for_stop_indices, label="variance")
plt.title('Mean and Variance of Kinematic Data for different stop indices')
plt.xlabel('Samples Included')
plt.ylabel('Mean and Variance')
plt.legend()
plt.show()

# %%
# plot kinematics for video 0
video_start_bin = 58153
video_len = 43 # seconds
video_end_bin = int(58153 + 43 * 1000 // dataset.bin_size)
use_smooth_data = False
all_kin_df_mine, rates_slice, body_part_names, orig_indices = return_all_nonNan_slice(use_smooth_data=use_smooth_data)
# %%
# video_bins = list(range(video_start_bin, video_end_bin+1))

# video_kinematics = dataset.data.kin_pos.iloc[video_start_bin:video_end_bin+1]

# # Find the intersection while preserving the order
# video_bins_with_kin = []
# video_bins_no_kin = []
# for bin_num in video_bins:
#     if bin_num in orig_indices:
#         video_bins_with_kin.append(bin_num)
#     else:
#         video_bins_no_kin.append(bin_num)
# Store the original DataFrame
original_kin_df = dataset.data.kin_pos

# Get the non-NaN slice
non_nan_slice = original_kin_df.dropna()

# Perform the operations on the non-NaN slice
non_nan_slice = append_kinematic_angle_data(non_nan_slice.copy())

# Combine the original DataFrame with the non-NaN slice
all_kin_and_angle_df = non_nan_slice.combine_first(original_kin_df)

# Create boolean masks for NaN values
ankle_x_nan = all_kin_and_angle_df["hip_x"].isna()
knee_angle_nan = all_kin_and_angle_df["knee_angle"].isna()



# %%
start_idx = 22000
stop_idx = 23000

ankle_angle = all_kin_and_angle_df["ankle_angle"].iloc[start_idx:stop_idx]
knee_angle = all_kin_and_angle_df["knee_angle"].iloc[start_idx:stop_idx]
ankle_x_pos = all_kin_and_angle_df['ankle_x'].iloc[start_idx:stop_idx]
rates_slice = rates_slice[start_idx: stop_idx]

assert(rates_slice.shape[0] == ankle_angle.shape[0])

fig, ax = plt.subplots(3,1, figsize=(5,10))
x = np.arange(start_idx, stop_idx)
y = np.arange(rates_slice.shape[1])  # assuming rates_slice.shape[1] is the correct dimension for your rates
X, Y = np.meshgrid(x, y)

# Use the generated X (2D array of x-coordinates) and Y (2D array of y-coordinates) for pcolor
vmax = np.max(rates_slice)
c = ax[0].pcolor(X, Y, rates_slice.T, cmap='viridis', vmin=0, vmax=vmax)
ax[0].set_title('Rates')
ax[0].set_xlabel('Time (bins)')
#plt.colorbar(c)
ax[0].set_title('Rates')
ax[0].set_xlabel('Time (bins)')
ax[0].set_ylabel('Rates')
ax[0].set_xlim(start_idx, stop_idx)



ax[1].plot(ankle_x_pos)
ax[1].set_title("Ankle X position")
ax[1].set_xlabel('Time (bins)')
ax[1].set_ylabel("Position (au)")

ax[2].plot(ankle_angle)
ax[2].set_title("Ankle angle")
ax[2].set_xlabel("Time (bins)")
ax[2].set_ylabel("Angle (degrees)")
fig.subplots_adjust(hspace=0.5)

plt.show()


# %%
