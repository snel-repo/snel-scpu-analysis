# %%
from snel_toolkit.datasets.nwb import NWBDataset
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import logging
import pandas as pd
import numpy as np
import sys
import os
# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%

base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
ds_name = "NP_AAV6-2_ReaChR_184500_kilosort.nwb"

ds_path = os.path.join(base_dir, ds_name)
logger.info(f"Loading {ds_name} from NWB")
ds = NWBDataset(ds_path, split_heldout=False)
# %%
BIN_WIDTH = 2 # ms
ds.resample(BIN_WIDTH)
# %%
event_id = 22
win_len_ms = 5000 # ms
pre_buffer_ms = 500 # ms
win_len = win_len_ms / BIN_WIDTH
pre_buff = pre_buffer_ms / BIN_WIDTH
event_start_time = ds.trial_info.iloc[event_id].start_time - pd.to_timedelta(pre_buff, unit="ms")
event_type = ds.trial_info.iloc[event_id].event_type
start_ix = ds.data.index.get_loc(event_start_time, method='nearest')
stop_ix = int(start_ix + win_len)

fig = plt.figure(figsize=(10,4), dpi=100)
time_vec = ds.data.index.values[start_ix:stop_ix+1].astype("timedelta64[ms]").astype(float)/1000.0
plt.pcolor(ds.data.spikes.values[start_ix:stop_ix,:].T, cmap=colormap.bone_r, vmin=0, vmax=2)

ax = plt.gca()
ax.set_title(f"{event_type}")
ax.vlines(pre_buff, 0, ds.data.spikes.shape[1], color="r")
ax.set_xticklabels(time_vec[ax.get_xticks().astype(int)])
ax.set_ylabel("neurons")
ax.set_xlabel("time (s)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# %%

start_chan_ix = 81
plot_n_chans = 9

width = np.ceil(np.sqrt(plot_n_chans)).astype(int)
height = np.floor(np.sqrt(plot_n_chans)).astype(int)
pre_offset_ms = -100
post_offset_ms = 200
stim_line_x = -1*(pre_offset_ms / BIN_WIDTH)
time_vec = np.arange(pre_offset_ms, post_offset_ms, BIN_WIDTH)
events = ds.trial_info[ds.trial_info.event_type == "stimulation"].trial_id.values
fig, axs = plt.subplots(width, height, figsize=(6,6), dpi=150)
axs = axs.flatten()
for i in range(plot_n_chans):
    trial_dat = []
    chan_ix = start_chan_ix + i
    if ds.data.spikes.columns.isin([chan_ix]).any():
        for event in events:        
            event_start_time = ds.trial_info.iloc[event].start_time + pd.to_timedelta(pre_offset_ms, unit="ms")    
            event_stop_time = ds.trial_info.iloc[event].start_time + pd.to_timedelta(post_offset_ms, unit="ms")    
            start_ix = ds.data.index.get_loc(event_start_time, method='nearest')
            stop_ix = ds.data.index.get_loc(event_stop_time, method='nearest')
            dat = ds.data.spikes.iloc[start_ix:stop_ix, chan_ix].values
            trial_dat.append(dat)
        ax = axs[i]
        ax.pcolor(np.array(trial_dat), cmap=colormap.bone_r, vmin=0, vmax=2)    
        ax.set_title(f"channel {chan_ix}", fontsize=8)
        ax.vlines(stim_line_x, 0, len(trial_dat), color="r")
        xticks = ax.get_xticks().astype(int)
        xticks = xticks*BIN_WIDTH + pre_offset_ms
        ax.set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
fig.tight_layout()
# %%
