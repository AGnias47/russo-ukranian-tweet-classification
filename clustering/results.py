#!/usr/bin/env python3

"""
Process results from clustering
"""

from utils.data import read_sample, write_sample
from utils.labels import assign_biclass_labels, color_dict

import matplotlib.pyplot as plt
import pandas as pd


def get_tweets_closest_to_centroid(clusters=3, data_directory="results/kmeans"):
    """
    Sorts tweets in order of their distance from their nearest centroid and writes the results to a file in the data
    directory

    Parameters
    ----------
    clusters: int
    data_directory: str (default is results/kmeans)

    Returns
    -------
    None
    """
    data = read_sample(f"{data_directory}/{clusters}/kmeans-output", cluster=True)
    for c in range(clusters):
        cluster = sorted(filter(lambda x: x.cluster == c, data), key=lambda x: x.cluster_distance)
        write_sample(f"{data_directory}/{clusters}/sorted_cluster_{c}.twint", cluster[:300], cluster=True)
    return None


def cluster_analysis(c1, c2, c3, biclass=True):
    """
    Creates a bar plot of clusters with their assigned labels

    Parameters
    ----------
    c1: Collection
        List of tweets corresponding to the first cluster
    c2: Collection
        List of tweets corresponding to the second cluster
    c3: Collection
        List of tweets corresponding to the third cluster
    biclass: bool
        If true, combine Pro-Ukraine and Anti-Russia Tweets,
        as well as Pro-Russia and Anti-Ukraine Tweets

    Returns
    -------
    None
    """
    if biclass:
        for s in c1 + c2 + c3:
            assign_biclass_labels(s)
    pd_table = pd.DataFrame()
    fig, ax = plt.subplots(1, 3, figsize=(8, 5), sharex=True, sharey=True)
    for i, c in enumerate([c1, c2, c3]):
        cluster = pd.DataFrame().append(
            pd.DataFrame([{"label": t.label} for t in c]).value_counts("label"),
            ignore_index=True,
        )
        cluster.plot.bar(
            stacked=True,
            ax=ax[i],
            title=f"Cluster {i} Tweets",
            xlabel=None,
            ylabel="Count",
            fontsize=12,
            legend=False,
            color=color_dict,
        ).xaxis.set_visible(False)
        if biclass:
            cluster_pu = cluster.iloc[0]["pro-ukraine"]
            cluster_au = cluster.iloc[0]["pro-russia"]
            cluster_neutral = cluster.iloc[0]["neutral"]
            ALT1 = -0.12
            ALT2 = -0.08
            total = len(c)
            if i == 2:
                ax[i].text(ALT1, cluster_pu / 2, f"{round(cluster_pu / total * 100, 2)}%", fontsize=12)
                ax[i].text(
                    ALT1, cluster_pu + cluster_neutral / 2, f"{round(cluster_neutral / total * 100, 2)}%", fontsize=12
                )
                ax[i].text(ALT2, total, f"{round(cluster_au / total * 100, 2)}%", fontsize=12)
            else:
                ax[i].text(ALT1, cluster_neutral / 2, f"{round(cluster_neutral / total * 100, 2)}%", fontsize=12)
                ax[i].text(
                    ALT1, cluster_neutral + cluster_pu / 2, f"{round(cluster_pu / total * 100, 2)}%", fontsize=12
                )
                ax[i].text(
                    ALT2,
                    cluster_pu + cluster_neutral + cluster_au / 2 - 7,
                    f"{round(cluster_au / total * 100, 2)}%",
                    fontsize=12,
                )
        else:
            cluster_pu = cluster.iloc[0]["pro-ukraine"]
            try:
                cluster_pr = cluster.iloc[0]["pro-russia"]
            except KeyError:
                cluster_pr = 0
            cluster_au = cluster.iloc[0]["anti-ukraine"]
            cluster_ar = cluster.iloc[0]["anti-russia"]
            cluster_neutral = cluster.iloc[0]["neutral"]
            culster_other_entity = cluster.iloc[0]["other-entity"]
            cluster_unknown = cluster.iloc[0]["unknown"]
            total = len(c)
            pd_table = pd_table.append(
                dict(
                    {
                        "Pro-Ukraine": (cluster_pu, round(cluster_pu / total * 100, 2)),
                        "Pro-Russia": (cluster_pr, round(cluster_pr / total * 100, 2)),
                        "Anti-Ukraine": (cluster_au, round(cluster_au / total * 100, 2)),
                        "Anti-Russia": (cluster_ar, round(cluster_ar / total * 100, 2)),
                        "Neutral": (cluster_neutral, round(cluster_neutral / total * 100, 2)),
                        "Other Entity": (culster_other_entity, round(culster_other_entity / total * 100, 2)),
                        "Unknown": (cluster_unknown, round(cluster_unknown / total * 100, 2)),
                    }
                ),
                ignore_index=True,
            )
    print(pd_table.to_latex())
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    if biclass:
        plt.savefig("final-report/images/biclass-kmeans.png")
    else:
        plt.savefig("final-report/images/quadclass-kmeans.png")
    plt.show()
    return None
