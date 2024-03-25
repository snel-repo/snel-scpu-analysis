"""
PURPOSE: Decode the latent factors using a linear decoder

REQUIREMENTS: merged pkl file from merge_chopped_torch_outputs.py
"""

# %% INPUTS AND PATHS
%load_ext autoreload
%autoreload 2
import os
import pickle as pkl
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from snel_toolkit.datasets.nwb import NWBDataset

import dill
from decoding_functions import *

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
load_smooth_data = True

path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)
smooth_data_path = "/snel/share/share/derived/scpu_snel/nwb_lfads/runs/binsize_2/NG_smooth_data.pkl"
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"

if load_smooth_data:
    with open(smooth_data_path,'rb') as inf:
        dataset = pkl.load(inf)
        bin_size = dataset.bin_size

        
else:
    gauss_width_ms = 25

    dataset, bin_size = load_dataset_and_binsize(yaml_config_path)
    dataset.smooth_spk(signal_type='kin_pos', gauss_width=3, name='smooth_3', overwrite=False)
    dataset.smooth_spk(signal_type='spikes', gauss_width=gauss_width_ms, name='smooth_25', overwrite=False)

    with open(smooth_data_path, "wb") as f:
        dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    # dataset has kinematic data, smoothed spikes, and smoothed lfads_factors

# %% 
#dataset.smooth_spk(signal_type='kin_pos', gauss_width=20, name='smooth_20', overwrite=False)
gauss_width_ms = 25
gauss_width_s = gauss_width_ms/1000
# cutoff frequency using -3db cutoff
cutoff_freq = (2 * np.sqrt(2 * np.log(2))) / (gauss_width_s*(np.sqrt(np.pi)))
print(f'Cutoff frequency: {cutoff_freq} Hz')
# %%
#dataset.smooth_spk(signal_type='lfads_rates', gauss_width=gauss_width_ms, name=f'smooth_{gauss_width_ms}', overwrite=False)
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
use_smooth_data = False


alpha_values = np.logspace(-5, 2, num=10)
use_velocity = True

r2_values_test_all = [] 
r2_values_train_all = []
train_sem_all = []

test_sem_all = [] 
fold = 10
r2_test_value_fold = []
r2_train_value_fold = []
# these are handpicked indices from the nonNan data. currently, they are based on the first video
# start_indices = [0] # this is orig_index 58153 and first video starts at 58027 meaning we have kinematics from 
# stop_indices = [5000] # end times for video in trial_info is wrong so can just choose some arbitrary end time
start_indices = [18000]
stop_indices = [22000+4000]
# start_indices = []
# stop_indices = []
for idx, column_name in enumerate(dataset.data.kin_pos.columns):
    if idx > 0:
        # only decode one column for now
        break
    column_name = "ankle_x"
    regression_vel_slice, regression_rates_slice, orig_indices = preprocessing(dataset, column_name, use_smooth_data, start_indices=start_indices, stop_indices=stop_indices, plot=True, use_LFADS=True, use_velocity=use_velocity)
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

# %%
velocity_title_modifier = 'Velocity' if use_velocity else 'Position'

# plot best alpha
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f"LFADS Rates indices: {orig_indices[0], orig_indices[-1]} decoding {column_name} {velocity_title_modifier}")
ax[0].plot(best_true, label='True')
ax[0].plot(best_pred, label='Predicted')
ax[0].set_title(f'Predicted vs True. Best Alpha: {round(best_alpha, 3)} Best Validation $R^2$: {round(best_r2, 6)}')
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
use_smooth_data = False
use_velocity = True
start_indices = [[22000], [22000], [22000], [22000], [22000], [22000], [22000], [22000], [22000], [22000], [22000, fill_in]]
stop_indices = [[22000+600],  [22000+1000],  [22000+1400],  [22000+1800], [22000+2200], [22000 + 2600], [22000 + 3000], [22000 + 3400], [22000 + 3800], [22000 + 4200], [22000 + 4600, fill_in]]
mean_kin_for_stop_indices = []
var_kin_for_stop_indices = []
num_batches = len(stop_indices)
num_channels = dataset.data.spikes.shape[1]
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
        
        regression_vel_slice, regression_rates_slice, orig_indices = preprocessing(dataset, column_name, use_smooth_data=use_smooth_data, start_indices=start_idx, stop_indices=stop_idx, plot=False, use_LFADS=use_lfads_param, use_velocity=use_velocity)

        for idx, alpha_x in enumerate(alpha_values):

            predicted_vel, r2_test, r2_train, train_sem, test_sem = linear_regression_train_val(regression_rates_slice, regression_vel_slice, alpha=alpha_x, folds=fold)

            # mean_kin_for_stop_indices.append(np.mean(regression_vel_slice))
            # var_kin_for_stop_indices.append(np.var(regression_vel_slice))

            # # rates distribution statistics

            # mean_rates_for_stop_indices[i, :] = np.mean(regression_rates_slice, axis=0)
            # var_rates_for_stop_indices[i, :] = np.var(regression_rates_slice, axis=0)
            if r2_test > best_r2:
                best_r2 = r2_test
                best_alpha = alpha_x
        if use_lfads_param:
            r2_across_dataset_sizes_LFADS.append(best_r2)
        else:
            r2_across_dataset_sizes_smooth.append(best_r2)

assert(len(r2_across_dataset_sizes_smooth) == len(stop_indices))
assert(len(r2_across_dataset_sizes_LFADS) == len(stop_indices))
# %% 
velocity_title_modifier = 'Velocity' if use_velocity else 'Position'
plt.figure()
plt.title(f'Change in Validation $R^2$ as more samples are included decoding {column_name} {velocity_title_modifier}\n$R^2$ chosen from L2 regularization sweep')
plt.plot(dataset_sizes, r2_across_dataset_sizes_smooth, '-o', label='Smoothed $R^2$')
plt.plot(dataset_sizes, r2_across_dataset_sizes_LFADS, '-o', label='LFADS R^2')
plt.xticks(dataset_sizes)
plt.xlabel('Sample Size')
plt.ylabel('$R^2$')
plt.legend()


plt.show()



# %%




# %%
# video timestamps 
# Walking begins ~2 seconds
# plot kinematics for video 0
video_start_bin = 58027
video_len = 43 # seconds
video_end_bin = int(video_start_bin + video_len * 1000 // dataset.bin_size)
use_smooth_data = False
column_name = 'ankle_x'
original_df = dataset.data.kin_pos
all_kin_df_mine, rates_slice, body_part_names = return_all_nonNan_slice(dataset, use_smooth_data=use_smooth_data)
# %%
non_nan_kin_with_angle = append_kinematic_angle_data(all_kin_df_mine)
all_kin_and_angle_df = non_nan_kin_with_angle.combine_first(original_df)
all_kin_and_angle_df.fillna(-1, inplace=True)

# !!combine_first is confirmed working!!


t_axis = np.arange(video_start_bin, video_end_bin)
fig, axs = plt.subplots(2, 1, figsize=(10, 4))
axs = axs.flatten()
fig.suptitle(f'Kinematic data from video 0')


# plot indices for first locomotion bout (indices 60000-62500)
start_idx = 66500
stop_idx = 68500
for idx, column_name in enumerate(['ankle_x', 'knee_angle']):
    kin_data_plot = all_kin_and_angle_df[column_name].iloc[start_idx:stop_idx]
    t_axis = np.arange(start_idx, stop_idx)
    t_axis -= video_start_bin
    # convert to seconds
    t_axis = t_axis * dataset.bin_size / 1000
    axs[idx].plot(t_axis, kin_data_plot, label=column_name)
    axs[idx].set_title(f'Indices: {start_idx, stop_idx} Data for {column_name} ')
    axs[idx].set_xlabel('Time (s) relative to video start')
    y_label = 'Position (au)' if column_name == 'ankle_x' else 'Angle (degrees)'
    axs[idx].set_ylabel(y_label)
    axs[idx].legend()

fig.subplots_adjust(hspace=1)




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
# plot ankle_x position over all time

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ankle_x_pos = dataset.data.kin_info_cw['mtarsal_x']
ankle_x_pos = ankle_x_pos.fillna(-1)
ankle_x_pos.index = np.arange(ankle_x_pos.shape[0])

ankle_x_pos[:500000].plot(ax=ax)
ax.set_title('Ankle X position over all time')
ax.set_xlabel('Time (bins)')
ax.set_ylabel('Position (au)')

# %%
