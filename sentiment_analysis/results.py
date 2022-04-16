#!/usr/bin/env python3

"""
Processes results for Sentiment Analysis
"""

from utils.labels import INT_TO_LONGFORM_DICT, color_dict, assign_biclass_labels

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rc("axes", labelsize=12)


def biclass_analysis(pro_ukraine_sample, anti_russia_sample, pro_russia_sample, anti_ukraine_sample, neutral_sample):
    """
    Analyzes results using Pro-Ukraine, Pro-Russia, and Neutral as categories

    Parameters
    ----------
    pro_ukraine_sample: Collection
    anti_russia_sample: Collection
    pro_russia_sample: Collection
    anti_ukraine_sample: Collection
    neutral_sample: Collection

    Returns
    -------
    None
    """
    for s in pro_ukraine_sample + anti_russia_sample:
        assign_biclass_labels(s)
    fig, ax = plt.subplots(1, 3, figsize=(8, 5), sharex=True, sharey=True)
    pu = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in pro_ukraine_sample + anti_russia_sample]).value_counts("label"),
        ignore_index=True,
    )
    pu.plot.bar(
        stacked=True,
        ax=ax[0],
        title="Expected Pro-Ukraine",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        legend=False,
        color=color_dict,
    ).xaxis.set_visible(False)
    pu_pu = pu.iloc[0]["pro-ukraine"]
    pu_au = pu.iloc[0]["pro-russia"]
    pu_n = pu.iloc[0]["neutral"]
    ALT1 = -0.12
    ALT2 = -0.08
    total = len(pro_ukraine_sample + anti_russia_sample)
    ax[0].text(ALT1, pu_pu / 2, f"{round(pu_pu / total * 100, 2)}%", fontsize=12)
    ax[0].text(ALT1, pu_pu + pu_n / 2, f"{round(pu_n / total * 100, 2)}%", fontsize=12)
    ax[0].text(ALT2, pu_pu + pu_n + pu_au / 2 - 7, f"{round(pu_au / total * 100, 2)}%", fontsize=12)
    for s in pro_russia_sample + anti_ukraine_sample:
        assign_biclass_labels(s)
    pr = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in pro_russia_sample + anti_ukraine_sample]).value_counts("label"),
        ignore_index=True,
    )
    pr.plot.bar(
        stacked=True,
        ax=ax[1],
        title="Expected Pro-Russia",
        xlabel=None,
        fontsize=12,
        legend=False,
        color=color_dict,
    ).xaxis.set_visible(False)
    pr_pu = pr.iloc[0]["pro-ukraine"]
    pr_pr = pr.iloc[0]["pro-russia"]
    pr_n = pr.iloc[0]["neutral"]
    total = len(pro_russia_sample + anti_ukraine_sample)
    ax[1].text(ALT1, pr_n / 2, f"{round(pr_n / total * 100, 2)}%", fontsize=12)
    ax[1].text(ALT1, pr_n + pr_pu / 2, f"{round(pr_pu / total * 100, 2)}%", fontsize=12)
    ax[1].text(ALT1, pr_n + pr_pu + pr_pr / 2, f"{round(pr_pr / total * 100, 2)}%", fontsize=12)
    for s in neutral_sample:
        assign_biclass_labels(s)
    n_df = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in neutral_sample]).value_counts("label"), ignore_index=True
    )
    n_df.plot.bar(
        stacked=True,
        ax=ax[2],
        title="Expected Neutral",
        xlabel=None,
        fontsize=12,
        legend=False,
        color=color_dict,
    ).xaxis.set_visible(False)
    n_pu = n_df.iloc[0]["pro-ukraine"]
    n_pr = n_df.iloc[0]["pro-russia"]
    n_n = n_df.iloc[0]["neutral"]
    total = len(neutral_sample)
    ax[2].text(ALT1, n_n / 2, f"{round(n_n / total * 100, 2)}%", fontsize=12)
    ax[2].text(ALT1, n_n + n_pu / 2, f"{round(n_pu / total * 100, 2)}%", fontsize=12)
    ax[2].text(ALT2, total, f"{round(n_pr / total * 100, 2)}%", fontsize=12)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower right")
    plt.savefig("final-report/images/biclass-sia.png")
    plt.show()
    return None


def quadclass_analysis(pro_ukraine_sample, anti_russia_sample, pro_russia_sample, anti_ukraine_sample, neutral_sample):
    """
    Analyzes results using all Desired Categories

    Parameters
    ----------
    pro_ukraine_sample: Collection
    anti_russia_sample: Collection
    pro_russia_sample: Collection
    anti_ukraine_sample: Collection
    neutral_sample: Collection

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    df_pu = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in pro_ukraine_sample]).value_counts("label"), ignore_index=True
    )
    df_pu.plot.bar(
        stacked=True,
        ax=ax[0, 0],
        title="Expected Pro-Ukraine Sentiment",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        color=color_dict,
        legend=False,
    ).xaxis.set_visible(False)
    df_ar = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in anti_russia_sample]).value_counts("label"), ignore_index=True
    )
    df_ar.plot.bar(
        stacked=True,
        ax=ax[0, 1],
        title="Expected Anti-Russia Sentiment",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        color=color_dict,
        legend=False,
    ).xaxis.set_visible(False)
    df_pr = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in pro_russia_sample]).value_counts("label"), ignore_index=True
    )
    df_pr.plot.bar(
        stacked=True,
        ax=ax[1, 0],
        title="Expected Pro-Russia Sentiment",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        color=color_dict,
        legend=False,
    ).xaxis.set_visible(False)
    df_au = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in anti_ukraine_sample]).value_counts("label"), ignore_index=True
    )
    df_au.plot.bar(
        stacked=True,
        ax=ax[1, 1],
        title="Expected Anti-Ukraine Sentiment",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        color=color_dict,
        legend=False,
    ).xaxis.set_visible(False)
    df_n = pd.DataFrame().append(
        pd.DataFrame([{"label": t.label} for t in neutral_sample]).value_counts("label"), ignore_index=True
    )
    df_n.plot.bar(
        stacked=True,
        ax=ax[2, 0],
        title="Expected Neutral Sentiment",
        xlabel=None,
        ylabel="Count",
        fontsize=12,
        color=color_dict,
        legend=False,
    ).xaxis.set_visible(False)
    ax[2, 1].axis("off")
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="lower right")
    plt.savefig("final-report/images/quadclass-sia.png")
    plt.show()
    class_labels = [INT_TO_LONGFORM_DICT[i + 1] for i in range(5)]
    y_actual = [
        [
            df_pu.iloc[0]["pro-ukraine"],
            0,
            0,
            df_pu.iloc[0]["anti-russia"],
            df_pu.iloc[0]["neutral"],
        ],
        [
            df_pr.iloc[0]["pro-ukraine"],
            df_pr.iloc[0]["pro-russia"],
            df_pr.iloc[0]["anti-ukraine"],
            df_pr.iloc[0]["anti-russia"],
            df_pr.iloc[0]["neutral"],
        ],
        [
            df_au.iloc[0]["pro-ukraine"],
            df_au.iloc[0]["pro-russia"],
            df_au.iloc[0]["anti-ukraine"],
            df_au.iloc[0]["anti-russia"],
            df_au.iloc[0]["neutral"],
        ],
        [
            0,
            df_ar.iloc[0]["pro-russia"],
            df_ar.iloc[0]["anti-ukraine"],
            df_ar.iloc[0]["anti-russia"],
            df_ar.iloc[0]["neutral"],
        ],
        [
            df_n.iloc[0]["pro-ukraine"],
            df_n.iloc[0]["pro-russia"],
            df_n.iloc[0]["anti-ukraine"],
            df_n.iloc[0]["anti-russia"],
            df_n.iloc[0]["neutral"],
        ],
    ]
    df_cm = pd.DataFrame(y_actual, index=class_labels, columns=class_labels)
    plt.figure(figsize=(12, 7))
    cf = sns.heatmap(df_cm, annot=True, cmap="flare")
    plt.savefig("final-report/images/quadclass-sia-cf.png")
    cf.set_xlabel("Predicted", fontsize=12)
    cf.set_ylabel("Actual", fontsize=12)
    plt.show()
