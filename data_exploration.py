# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import os
import pandas as pd
# %% Load data
base_dir = "/snel/share/share/data/scpu_data/"
spike_clusters = np.load(os.path.join(base_dir, "spike_clusters.npy"))
spike_samp_nums = np.load(os.path.join(base_dir, "spike_times.npy"))
cluster_info = pd.read_csv(os.path.join(base_dir, "cluster_info.tsv"), sep="\t")
sampling_rate = 30000 # Hz
 
spike_times = spike_samp_nums / sampling_rate

# %% Create a unique list of cluster IDs
cluster_ids = np.unique(spike_clusters)

# %% Create an empty matrix to store the spike trains
max_spike_time_s = int(np.max(spike_times))
bin_size_s = 0.001 # 10 ms bins
bins = np.arange(0, max_spike_time_s+ bin_size_s, bin_size_s) # 1 ms bins
trains_matrix = np.zeros((bins.size-1, len(cluster_ids)))

# %% Fill the matrix with spike train data
n_spikes_per_cluster = []
clust_depths = []
for i, clust_id in enumerate(cluster_ids):
    clust_info_mask = cluster_info.cluster_id == clust_id
    clust_depth = cluster_info.loc[clust_info_mask].depth
    cluster_times = spike_times[spike_clusters == clust_id]
    binned_spikes, _ = np.histogram(cluster_times, bins=bins)
    trains_matrix[:, i] = binned_spikes
    n_spikes_per_cluster.append(cluster_times.size)
    clust_depths.append(clust_depth)


# %% Plot the spike trains 

# plot params
plot_n_chans = 588 # number of channels to plot
plot_start_s = 100 # seconds 
plot_end_s = 200 # seconds 

plot_order = np.argsort(n_spikes_per_cluster)

plot_order = np.arange(len(cluster_ids))
plot_chans = plot_order[-plot_n_chans:]
plot_start = int(plot_start_s*(1/bin_size_s))
plot_end = int(plot_end_s*(1/bin_size_s))
fig = plt.figure(figsize=(10,4), dpi=200)
ax = fig.add_subplot(111)
ax.pcolor(trains_matrix[plot_start:plot_end, plot_chans].T, vmin=0, vmax=1, cmap=colormap.binary)
ax.set_xticklabels(ax.get_xticks()*bin_size_s)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cluster (sorted)')
ax.set_title('Spike Trains by Cluster')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()

# %% Plot the spike trains by depth
plot_n_chans = 588 # number of channels to plot
plot_start_s = 100 # seconds 
plot_end_s = 200 # seconds 


plot_chans = plot_order[-plot_n_chans:]
plot_start = int(plot_start_s*(1/bin_size_s))
plot_end = int(plot_end_s*(1/bin_size_s))
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)

depth_cutoff = 1200
for i, depth in enumerate(clust_depths):
    if depth.values < depth_cutoff:
        chan_data = trains_matrix[:, i]
        spk_tstamps = np.where(chan_data == 1)[0]
        spk_times = bins[spk_tstamps]
        depths = np.ones_like(spk_times)*depth.values
        ax.scatter(spk_times, depths, s=0.2, alpha=0.1, c='k')

ax.set_xlim([plot_start_s, plot_end_s])
ax.set_ylim([0, depth_cutoff])
ax.set_ylabel("y position ($\mu$m)")
ax.set_xlabel("Time (s)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# %%
