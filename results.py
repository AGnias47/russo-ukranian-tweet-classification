#!/usr/bin/env python3

from utils.data import read_sample
from utils.labels import label_longform
from sentiment_analysis.results import biclass_analysis, quadclass_analysis
from clustering.results import cluster_analysis
from nn.results import run_via_holdout, run_via_cross_validation, run_via_cross_validation_augmented
from nn.data import load_data

from copy import deepcopy

import torch.optim


def sia_results():
    """
    Process Sentiment Analysis results

    Returns
    -------
    None
    """
    data_path = "data/labelled/sia"
    pro_ukraine_sample = read_sample(f"{data_path}/pro_ukraine.twint", label=True)
    anti_russia_sample = read_sample(f"{data_path}/anti_russia.twint", label=True)
    pro_russia_sample = read_sample(f"{data_path}/pro_russia.twint", label=True)
    anti_ukraine_sample = read_sample(f"{data_path}/anti_ukraine.twint", label=True)
    neutral_sample = read_sample(f"{data_path}/neutral.twint", label=True)
    for s in pro_ukraine_sample + anti_russia_sample + pro_russia_sample + anti_ukraine_sample + neutral_sample:
        s.label = label_longform(s.label)
    biclass_analysis(
        deepcopy(pro_ukraine_sample),
        deepcopy(anti_russia_sample),
        deepcopy(pro_russia_sample),
        deepcopy(anti_ukraine_sample),
        deepcopy(neutral_sample),
    )
    quadclass_analysis(pro_ukraine_sample, anti_russia_sample, pro_russia_sample, anti_ukraine_sample, neutral_sample)
    return None


def clustering_results():
    """
    Process clustering results

    Returns
    -------
    None
    """
    data_path = "data/labelled/cluster"
    cluster_0 = read_sample(f"{data_path}/sorted_cluster_0.twint", label=True)
    cluster_1 = read_sample(f"{data_path}/sorted_cluster_1.twint", label=True)
    cluster_2 = read_sample(f"{data_path}/sorted_cluster_2.twint", label=True)
    for s in cluster_0 + cluster_1 + cluster_2:
        s.label = label_longform(s.label)
    cluster_analysis(deepcopy(cluster_0), deepcopy(cluster_1), deepcopy(cluster_2), biclass=True)
    cluster_analysis(cluster_0, cluster_1, cluster_2, biclass=False)
    return None


def nn_results():
    """
    Process Neural Network results

    Returns
    -------
    None
    """
    sample = load_data(["data/labelled/sia", "data/labelled/cluster"])
    run_via_cross_validation(
        deepcopy(sample),
        input_layer=21,
        output_layer=21,
        lr=0.07355149846490083,
        gamma=0.013139089497364644,
        optim=torch.optim.RMSprop,
        classes=3,
    )
    sample = load_data(["data/labelled/sia", "data/labelled/cluster"])
    run_via_cross_validation_augmented(
        deepcopy(sample),
        input_layer=27,
        output_layer=5,
        lr=0.08280576428392994,
        gamma=0.0039336682124702605,
        optim=torch.optim.Adam,
        classes=3,
    )
    run_via_holdout(
        deepcopy(sample),
        input_layer=16,
        output_layer=20,
        lr=0.08512789475335665,
        gamma=0.05161381851509345,
        optim=torch.optim.RMSprop,
        classes=5,
    )
    run_via_holdout(
        deepcopy(sample),
        input_layer=29,
        output_layer=24,
        lr=0.0025844798144657464,
        gamma=0.0017753246663449508,
        optim=torch.optim.Adam,
        classes=7,
    )
    return None


if __name__ == "__main__":
    sia_results()
    clustering_results()
    nn_results()
