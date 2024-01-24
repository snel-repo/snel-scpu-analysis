"""
PURPOSE: 1.) Plot rasters of smoothed spikes, lfads inferred rates, and original spikes for a single trial
         2.) Plot rasters of smoothed spikes and lfads inferred rates for all stimulation trials for a single channel

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

# %% load YAML file and attributes

yaml_config_path = "../configs/lfads_dataset_cfg.yaml"

path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)

# system inputs
run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
initials = path_config["INITIALS"] # Ex: "cw"
run_type = path_config["TYPE"] # Ex: "spikes"
chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
bin_size = ld_cfg["BIN_SIZE"] # Ex: 2


# %% load dataset

ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
run_base_dir = f"/snel/share/runs/aav_{run_type}/lfads_{ds_name}/{run_date}_aav_{run_type}_PBT_{initials}"
run_dir = os.path.join(run_base_dir,"best_model")

merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
with open(merged_full_output, "rb") as f:
    dataset = dill.load(f)

# %% all channels raster for single trial
event_id = 25 # locomotion/stim trial idx
# LOCOMOTION: 1, 2*, 4* good; 0, 3 okay | 
# STIM: all but 19  | 18, 23, 24, 25/26/27 (periodic after) good examples

pre_buffer_ms = 500 # ms
win_len_ms = 4000 # ms
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
    # exception thrown when win len not divisible by 200 because get_xticks overshoots len of data
    try:
        axs[i].set_xticklabels(time_vec[axs[i].get_xticks().astype(int)])
    except Exception as e:
        axs[i].set_xticklabels(time_vec[axs[i].get_xticks().astype(int)[:-1]])
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


# %%
