"""
PURPOSE: Fit PCs on the latent factors and plot the PCs in 3D space, coloring by step cycles

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% load dataset and bin size

yaml_config_path = "../configs/lfads_dataset_cfg.yaml"

dataset, bin_size = load_dataset_and_binsize(yaml_config_path)

# %% prepare data to compute PCs

# use pandas to replace all NaN values in lfads_factors with 0 inplace
dataset.data.lfads_factors = dataset.data.lfads_factors.fillna(0)

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


# %%  visualize activation in channel 68 with threshold for "active" step cycles

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