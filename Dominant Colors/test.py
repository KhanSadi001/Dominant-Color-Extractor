import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, whiten
import pandas as pd
import seaborn as sns

# Read image
image = img.imread('sunset.jpg')

# Separate channels
r, g, b = [], [], []
for row in image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

# Get original std deviations (before whitening)
r_std_orig = np.std(r)
g_std_orig = np.std(g)
b_std_orig = np.std(b)

# Whiten each channel
r_w = whiten(r)
g_w = whiten(g)
b_w = whiten(b)

# Combine into DataFrame
df = pd.DataFrame({'r': r_w, 'g': g_w, 'b': b_w})

# Elbow method
distortions = []
num_clusters = range(1, 7)
for i in num_clusters:
    _, distortion = kmeans(df[['r', 'g', 'b']], i)
    distortions.append(distortion)

# Plot elbow
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()

# Choose k (from elbow)
k = 3
cluster_centers, _ = kmeans(df[['r', 'g', 'b']], k)

# Scale back from whitened to original RGB scale
colors = []
for scaled_r, scaled_g, scaled_b in cluster_centers:
    colors.append((
        scaled_r * r_std_orig / 255,  # scale to 0–1
        scaled_g * g_std_orig / 255,
        scaled_b * b_std_orig / 255
    ))

# Show palette
plt.imshow([colors])
plt.axis('off')
plt.show()
