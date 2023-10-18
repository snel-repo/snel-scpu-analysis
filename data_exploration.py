# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Load data
spike_clusters = np.load("data/spike_clusters.npy")
spike_times = np.load("data/spike_times.npy")

# %% Create a unique list of cluster IDs
cluster_ids = np.unique(spike_clusters)

# %% Create an empty matrix to store the spike trains
trains_matrix = np.zeros((len(cluster_ids), int(np.max(spike_times)) + 1))

# %% Fill the matrix with spike train data
for i, cluster in enumerate(cluster_ids):
    cluster_times = spike_times[spike_clusters == cluster]
    trains_matrix[i, cluster_times] = 1


# %% Find region of dense spikes for initial plot (13311000)
scale_factor = 1000
# partial_sums = []
# for i in range(int(np.ceil(len(trains_matrix[0])/scale_factor))):
#     try:
#         partial_sums.append(np.sum(trains_matrix[:,scale_factor*i:scale_factor*(i+1)]))
#     except:
#         partial_sums.append(np.sum(trains_matrix[:,scale"_factor*i:]))
# dense_spike_start = scale_factor*np.argmax(partial_sums)

dense_spike_start = 13311000 #for initial data


# %% Plot the spike trains 
plt.imshow(trains_matrix[:,dense_spike_start:dense_spike_start+scale_factor], cmap=plt.cm.binary, aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Cluster')
plt.title('Spike Trains by Cluster')
plt.show()


# %%
