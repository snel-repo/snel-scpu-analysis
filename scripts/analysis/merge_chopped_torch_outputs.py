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

# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)

path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg["DATASET"]
merge_config = lfads_dataset_cfg["MERGE_PARAMETERS"]


# system inputs
run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
initials = path_config["INITIALS"] # Ex: "cw"
run_type = path_config["TYPE"] # Ex: "spikes"
chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
bin_size = ld_cfg["BIN_SIZE"] # Ex: 2

# create paths
ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
run_base_dir = f"/snel/share/runs/aav_{run_type}/lfads_{ds_name}/{run_date}_aav_{run_type}_PBT_{initials}"
run_dir = os.path.join(run_base_dir,"best_model")
lfads_torch_outputs_path = os.path.join(run_dir,f"lfads_output_lfads_{ds_name}_out.h5")
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"
unchopped_ds_path = os.path.join(lfads_save_dir,"lfads_"+ds_name+"_unchopped.pkl")
interface_path = f"{lfads_save_dir}pkls/{ds_name}_interface.pkl"
DATA_FILE = os.path.join(lfads_save_dir, ds_name)

cache_dataset = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_fulldataset.pkl"

original_h5 = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}.h5"


# %% LOAD CONTINUOUS DATA DF, MERGE WITH TORCH OUTPUTS  
with open(interface_path,'rb') as inf:
    interface = pkl.load(inf)

interface.merge_fields_map = merge_config

with open(cache_dataset,'rb') as inf:
    dataset = pkl.load(inf)

torch_outputs = h5py.File(lfads_torch_outputs_path)

# %% Load chop indices pertaining to training and validation; add to torch output obj if not present

train_inds, valid_inds = get_train_valid_inds(original_h5, torch_outputs, lfads_torch_outputs_path)

# %% Make full output df

data_dict = combine_train_valid_outputs(torch_outputs, train_inds, valid_inds, merge_config)
merged_df = interface.merge(data_dict, smooth_pwr=1)

# %% Merge with original dataset
merge_with_original_df(merged_df, dataset)
# %% smooth spikes, rates, factors
# fill na with 0 for lfads outputs due to chopping
dataset.smooth_spk(gauss_width = 15, name='smooth_15', overwrite=False)

dataset.smooth_spk(signal_type='lfads_rates', gauss_width=8, name='smooth_8', overwrite=False)
dataset.data.lfads_rates_smooth_8 = dataset.data.lfads_rates_smooth_8.fillna(0)

dataset.smooth_spk(signal_type='lfads_factors', gauss_width=8, name='smooth_15', overwrite=False)
dataset.data.lfads_factors_smooth_15 = dataset.data.lfads_factors_smooth_15.fillna(0)

# %%
# save dataset to pickle
merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
with open(merged_full_output, "wb") as f:
    dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

# %% all channels raster for single trial
event_id = 25 # locomotion/stim trial idx
# LOCOMOTION: 1, 2*, 4* good; 0, 3 okay | 
# STIM: all but 19  | 18, 23, 24, 25/26/27 (periodic after) good examples



pre_buffer_ms = 200 # ms
win_len_ms = 1100 # ms
pre_buff = pre_buffer_ms / bin_size

start_ix, stop_ix = get_event_start_stop_ix(win_len_ms=win_len_ms, pre_buffer_ms=pre_buffer_ms, event_id=event_id, dataset=dataset)
event_type = dataset.trial_info.iloc[event_id].event_type
# make a 2 by 1 subplot
fig,axs = plt.subplots(3,1,figsize=(8,11),dpi=100)
plt.subplots_adjust(top=0.88)

# fig,axs = plt.subplot(2,1,figsize=(10,4),dpi=100)
# fig = plt.figure(figsize=(10,4), dpi=100)
smooth_slice = dataset.data.spikes_smooth_15.values[start_ix:stop_ix,:]
lfads_slice = dataset.data.lfads_rates.values[start_ix:stop_ix,:]
spikes_slice = dataset.data.spikes.values[start_ix:stop_ix,:]
vmin = 0 
vmax = max([smooth_slice.max(), lfads_slice.max()])
max_scale = 0.5
time_vec = dataset.data.index.values[start_ix:stop_ix+1].astype("timedelta64[ms]").astype(float)/1000.0

# fig.suptitle(f"{event_type} trial {event_id}", x=0.1,y=1)
# axs[0].pcolor(smooth_slice.T, cmap='viridis', vmin=vmin, vmax=max_scale*vmax)
# axs[1].pcolor(lfads_slice.T, cmap='viridis', vmin=vmin, vmax=max_scale*vmax)
# axs[2].pcolor(spikes_slice.T, cmap=colormap.bone_r, vmin=0, vmax=1)

# axs[0].set_title("smoothed spikes")
# axs[0].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
# axs[0].set_xticklabels(time_vec[axs[0].get_xticks().astype(int)])
# axs[0].set_ylabel("neurons")
# axs[0].set_xlabel("time (s)")
# axs[0].spines['right'].set_visible(False)
# axs[0].spines['top'].set_visible(False)

# axs[1].set_title("LFADS inferred rates")
# axs[1].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
# axs[1].set_xticklabels(time_vec[axs[1].get_xticks().astype(int)])
# axs[1].set_ylabel("neurons")
# axs[1].set_xlabel("time (s)")
# axs[1].spines['right'].set_visible(False)
# axs[1].spines['top'].set_visible(False)

# axs[2].set_title("original spikes")
# axs[2].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
# axs[2].set_xticklabels(time_vec[axs[2].get_xticks().astype(int)])
# axs[2].set_ylabel("neurons")
# axs[2].set_xlabel("time (s)")
# axs[2].spines['right'].set_visible(False)
# axs[2].spines['top'].set_visible(False)

# plt.subplots_adjust(hspace=0.5)
# fig.tight_layout()
# plt.show()

titles = ["smoothed spikes", "LFADS inferred rates", "original spikes"]
data_slices = [smooth_slice, lfads_slice, spikes_slice]
cmaps = ['viridis', 'viridis', colormap.bone_r]
vmaxs = [max_scale*vmax, max_scale*vmax, 1]
vmin = 0
fig.suptitle(f"{event_type} trial {event_id}", x=0.1, y=1)

for i in range(3):
    axs[i].pcolor(data_slices[i].T, cmap=cmaps[i], vmin=vmin, vmax=vmaxs[i])
    axs[i].set_title(titles[i])
    axs[i].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
    axs[i].set_xticklabels(time_vec[axs[i].get_xticks().astype(int)])
    axs[i].set_ylabel("neurons")
    axs[i].set_xlabel("time (s)")
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

plt.subplots_adjust(hspace=0.5)
fig.tight_layout()
plt.show()

# %%
chans = [34, 68, 85, 47, 79]
# start_chan_ix = 34

pre_offset_ms = -100
post_offset_ms = 200
stim_line_x = -1*(pre_offset_ms / bin_size)
time_vec = np.arange(pre_offset_ms, post_offset_ms, bin_size)
events = dataset.trial_info[dataset.trial_info.event_type == "stimulation"].trial_id.values

fig, axs = plt.subplots(2, len(chans), figsize=(10,3.5), dpi=200)
# Create a big subplot
big_ax = fig.add_subplot(111, frameon=False)
# Hide tick and tick label of the big subplot
big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
big_ax.grid(False)

# Set the labels
big_ax.set_xlabel('Time (ms) relative to stim. onset', labelpad=5)
big_ax.set_ylabel('Stim. trial #', labelpad=-15)

axs = axs.flatten()

for i, chan in enumerate(chans):
    trial_dat_spikes = []
    trial_dat_lfads = []
    if dataset.data.spikes.columns.isin([chan]).any():
        for event in events:        
            event_start_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(pre_offset_ms, unit="ms")    
            event_stop_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(post_offset_ms, unit="ms")    
            start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
            stop_ix = dataset.data.index.get_loc(event_stop_time, method='nearest')
            dat_spikes = dataset.data.spikes.iloc[start_ix:stop_ix, chan].values
            trial_dat_spikes.append(dat_spikes)


            dat_lfads = dataset.data.lfads_rates.iloc[start_ix:stop_ix, chan].values
            trial_dat_lfads.append(dat_lfads)
        axs[i].pcolor(np.array(trial_dat_spikes), cmap=colormap.bone_r, vmin=0, vmax=2)    
        axs[i].set_title(f"channel {chan}", fontsize=8)
        axs[i].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[i].get_xticks().astype(int)
        xticks = xticks*bin_size + pre_offset_ms
        axs[i].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i+len(chans)].pcolor(np.array(trial_dat_lfads), cmap='viridis', vmin=0, vmax=np.max(np.array(trial_dat_lfads)))    
        axs[i+len(chans)].set_title(f"channel {chan}", fontsize=8)
        axs[i+len(chans)].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[i+len(chans)].get_xticks().astype(int)
        xticks = xticks*bin_size + pre_offset_ms
        axs[i+len(chans)].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[i+len(chans)].spines['right'].set_visible(False)
        axs[i+len(chans)].spines['top'].set_visible(False)
    plt.plot()
fig.tight_layout()

# %% prepare data to compute PCs

# use pandas to replace all NaN values in lfads_factors with 0 inplace
dataset.data.lfads_factors = dataset.data.lfads_factors.fillna(0)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaled_factors = scaler.fit_transform(dataset.data.lfads_factors_smooth_15)

# %% SUBSET DATA
# select 1s before and 10s after each trial
trial_type = "locomotion"
# trial_type = "stimulation"

pre_offset_ms = 200
post_offset_ms = 1000
stim_line_x = -1*(pre_offset_ms / bin_size)
time_vec = np.arange(pre_offset_ms, post_offset_ms, bin_size)
events = dataset.trial_info[dataset.trial_info.event_type == trial_type].trial_id.values

scaled_factors_subset = []
start_stop_ix = []
for event in events:        
    event_start_time = dataset.trial_info.iloc[event].start_time - pd.to_timedelta(pre_offset_ms, unit="ms")    
    event_stop_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(post_offset_ms, unit="ms")    
    start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
    stop_ix = dataset.data.index.get_loc(event_stop_time, method='nearest')
    subset_data = scaled_factors[start_ix:stop_ix, :]
    scaled_factors_subset.append(subset_data)
    start_stop_ix.append((start_ix, stop_ix))
# %% compute PCs and plot variance explained

n_components = 40

pca = PCA(n_components=n_components)

scaled_factor_PCs = pca.fit_transform(scaled_factors)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(1, pca.explained_variance_ratio_.shape[0]+1),
        y=np.cumsum(pca.explained_variance_ratio_),
        mode='lines+markers'
    )
)
fig.update_layout(
    xaxis_title='PC',
    yaxis_title='Variance explained',
    title="PCA on latent factors: full dataset"
)
fig.show()

scaled_factors_subset_PCs = pca.fit_transform(np.concatenate(scaled_factors_subset, axis=0))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(1, pca.explained_variance_ratio_.shape[0]+1),
        y=np.cumsum(pca.explained_variance_ratio_),
        mode='lines+markers'
    )
)
fig.update_layout(
    xaxis_title='PC',
    yaxis_title='Variance explained',
    title="PCA on latent factors: locomotion trials only"
)
fig.show()

# %% fill in PCs with Nans to be full size of dataset

# full time for dataset with Nans where we did not do PCA
time_length = scaled_factors.shape[0]
assert time_length == 1814001
scaled_factors_subset_PCs_full_size = np.full((scaled_factors.shape[0], n_components), np.nan)

for i, (start_ix, stop_ix) in enumerate(start_stop_ix):
    scaled_factors_subset_PCs_full_size[start_ix:stop_ix, :] = scaled_factors_subset_PCs[i * (stop_ix - start_ix): (i + 1) * (stop_ix - start_ix),:]


# %% find which time points have factors with NaNs
# time is dim 0 of the factors array

# nan_mask = np.isnan(lfads_factors)
# nan_mask = np.any(nan_mask, axis=1)
# nan_ix = np.where(nan_mask)[0]
# nan_time = dataset.data.index.values[nan_ix]


# %%  visualize activation in channel 68 with threshold for "active" step cycles

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# create a 3D continuous plot of the first 3 PCs
neuron = 68 # 68 good for locomotion, 65 good for stim
event_id = 4 # locomotion/stim trial idx
# LOCOMOTION: 1, 2*, 4* good; 0, 3 okay | 
# STIM: all but 19  | 18, 23, 24, 25/26/27 (periodic after) good examples


win_len_ms = 1100 # ms
pre_buffer_ms = 200 # ms
win_len = win_len_ms / bin_size
pre_buff = pre_buffer_ms / bin_size
event_start_time = dataset.trial_info.iloc[event_id].start_time - pd.to_timedelta(pre_buffer_ms, unit="ms")
event_type = dataset.trial_info.iloc[event_id].event_type
start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
stop_ix = int(start_ix + win_len)

print("ms range:",win_len_ms)

fr_thresh = 0.025

# plot 68 itself
plt.title("channel {}".format)
plt.plot(dataset.data.lfads_rates_smooth_8.iloc[start_ix:stop_ix,neuron])
#add horizontal line at threshold
plt.axhline(fr_thresh, color='r')
plt.show()
# make mask for points above threshold
def make_mask(mask_type, data, thresh, start_ix, stop_ix):
    if mask_type == 'binary':
        mask = data.iloc[start_ix:stop_ix,neuron] > thresh
        mask = np.array(mask).astype(int)
    elif mask_type == 'segment_high':
        mask = np.full((stop_ix-start_ix),0)
        mask_diff=np.diff(np.insert(np.array(data.iloc[start_ix:stop_ix,neuron] > thresh).astype(int), 0,0),axis=0)
        onsets = np.where(mask_diff == 1)[0]
        offsets = np.where(mask_diff == -1)[0]
        offsets = np.insert(offsets,offsets.shape[0],stop_ix-start_ix)
        for i, (onset, offset) in enumerate(zip(onsets, offsets)):
            mask[onset:offset] = i+1
    return mask

# mask_type = 'binary'
mask_type= 'segment_high'
fr_mask = make_mask(mask_type, dataset.data.lfads_rates_smooth_8, \
                    fr_thresh, start_ix, stop_ix)
#m
# plot 68 diff
# plt.title("channel 68 diff")
# plt.plot(np.diff(dataset.data.lfads_rates_smooth_8.iloc[start_ix:stop_ix,68]))
# plt.show()

# %% create 3D plotly state space plots of scaled factor PCs 

PCs = scaled_factor_PCs
pc_fit_type = "all data"
# PCs = scaled_factors_subset_PCs_full_size
# pc_fit_type = "time around locomotion trials only"


fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=PCs[start_ix:stop_ix, 0], 
        y=PCs[start_ix:stop_ix, 1], 
        z=PCs[start_ix:stop_ix, 2],
        mode='lines',
        line=dict(
            color=fr_mask,
            colorscale='oxy',
            width=6
        )
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    title="LFADS factor PCs | PC space fit on {} | Trial {} | {} ms".format(pc_fit_type,event_id, win_len_ms),
)

#fix aspect ratio
fig.update_layout(scene_aspectmode='cube')

fig.show()








# %% make the same plot as above, but as a video that traces the path of the line
# first, make a 3D scatter plot of the points
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=scaled_factor_PCs[start_ix:stop_ix, 0], 
        y=scaled_factor_PCs[start_ix:stop_ix, 1], 
        z=scaled_factor_PCs[start_ix:stop_ix, 2],
        mode='markers',
        marker=dict(
            color=fr_mask,
            colorscale='oxy',
            size=4
        )
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    title="LFADS factor PCs | Trial {} | {} ms".format(event_id, win_len_ms),
)

#fix aspect ratio
fig.update_layout(scene_aspectmode='cube')

# now, add a point that traces the path of the line


fig.show()




# %% 2D version of above
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=scaled_factor_PCs[start_ix:stop_ix, 0], 
        y=scaled_factor_PCs[start_ix:stop_ix, 1], 
        mode='lines',
        line=dict(
            # color=np.arange(start_ix, stop_ix),
            # colorscale='viridis',
            width=2
        )
    )
)

fig.update_layout(
    xaxis_title='PC1',
    yaxis_title='PC2',
    title="LFADS factor PCs"
)

fig.show()



# %% 
