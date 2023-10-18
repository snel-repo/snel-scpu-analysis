# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Load data
spike_clusters = np.load("../spike_clusters.npy")
spike_times = np.load("../spike_times.npy")

# %% Create a unique list of cluster IDs
cluster_ids = np.unique(spike_clusters)

# %% Create an empty matrix to store the spike trains
trains_matrix = np.zeros((len(cluster_ids), int(np.max(spike_times)) + 1))

# %% Fill the matrix with spike train data
for i, cluster in enumerate(cluster_ids):
    cluster_times = spike_times[spike_clusters == cluster]
    trains_matrix[i, cluster_times] = 1

# %% Plot the spike trains
plt.imshow(trains_matrix[:,13311000:13312000], cmap=plt.cm.binary, aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Cluster')
plt.title('Spike Trains by Cluster')
plt.show()

# %%
partial_sums = []
for i in range(int(np.ceil(len(trains_matrix[0])/1000))):
    try:
        partial_sums.append(np.sum(trains_matrix[:,1000*i:1000*(i+1)]))
    except:
        partial_sums.append(np.sum(trains_matrix[:,1000*i:]))
np.argmax(partial_sums)
# %%
