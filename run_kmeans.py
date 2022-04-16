#!/usr/bin/env python3

from clustering.kmeans import kmeans_clustering, write_relevant_features_to_file
from utils.data import read_sample, write_sample

from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COUNT = 5

"""
Used to perform cluster analysis
"""

COLORMAP = "viridis"

if __name__ == "__main__":
    cluster_sizes = [3]  # , 5, 7, 15, 25]
    data_directory = "data"
    fname = "kmeans-input"
    sample = read_sample(f"{data_directory}/{fname}.twint")
    print(f"Performing cluster analysis on tweets from {fname}")
    for k in cluster_sizes:
        results_directory = f"results/kmeans/{k}"
        k_sample = deepcopy(sample)
        clusters, feature_dict, cluster_dict, pca_centroids, pca_tweets, distances = kmeans_clustering(k_sample, k)
        for cluster, tweet, distance in zip(clusters, k_sample, distances):
            tweet.cluster = cluster
            tweet.cluster_distance = distance
        write_sample(f"{results_directory}/kmeans-output", k_sample, cluster=True)
        write_relevant_features_to_file(f"{results_directory}/features", k, cluster_dict, feature_dict)
        scatterplot = sns.scatterplot(
            x=pca_tweets[:, 0], y=pca_tweets[:, 1], hue=clusters, s=5, alpha=0.7, palette=COLORMAP
        )
        scatterplot = sns.scatterplot(
            x=pca_centroids[:, 0],
            y=pca_centroids[:, 1],
            hue=range(k),
            s=25,
            ax=scatterplot,
            palette=COLORMAP,
            legend=False,
        )
        text_placement = [(0.0, 0.25), (0.05, 0.0), (-0.2, -0.02)]
        for i in range(k):
            content = "("
            j = 0
            for key, v in cluster_dict[i].items():
                if j > 0:
                    content += ", "
                content += feature_dict[key]
                j += 1
                if j >= 3:
                    content += ")"
                    break
            plt.text(s=content, x=text_placement[i][0], y=text_placement[i][1], fontsize=7)
        plt.savefig(f"final-report/images/kmeans-{k}.png")
        plt.show()
