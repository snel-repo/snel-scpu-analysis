# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
# %% Load data
spike_clusters = np.load("/snel/share/share/data/scpu_data/spike_clusters.npy")
spike_samp_nums = np.load("/snel/share/share/data/scpu_data/spike_times.npy")

sampling_rate = 30000 # Hz
 
spike_times = spike_samp_nums / sampling_rate

# %% Create a unique list of cluster IDs
cluster_ids = np.unique(spike_clusters)

# %% Create an empty matrix to store the spike trains
max_spike_time_s = int(np.max(spike_times))
bin_size_s = 0.010 # 10 ms bins
bins = np.arange(0, max_spike_time_s+ bin_size_s, bin_size_s) # 1 ms bins
trains_matrix = np.zeros((bins.size-1, len(cluster_ids)))

# %% Fill the matrix with spike train data
n_spikes_per_cluster = []
for i, cluster in enumerate(cluster_ids):
    cluster_times = spike_times[spike_clusters == cluster]
    binned_spikes, _ = np.histogram(cluster_times, bins=bins)
    trains_matrix[:, i] = binned_spikes
    n_spikes_per_cluster.append(cluster_times.size)


# %% Plot the spike trains 

# plot params
plot_n_chans = 300 # number of channels to plot
plot_len_s = 480 # seconds 

plot_order = np.argsort(n_spikes_per_cluster)
plot_chans = plot_order[-plot_n_chans:]
plot_len = int(plot_len_s*(1/bin_size_s))
fig = plt.figure(figsize=(10,4), dpi=200)
ax = fig.add_subplot(111)
ax.pcolor(trains_matrix[:plot_len, plot_chans].T, vmin=0, vmax=1, cmap=colormap.binary)
ax.set_xticklabels(ax.get_xticks()*bin_size_s)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cluster (sorted)')
ax.set_title('Spike Trains by Cluster')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()


# %%
