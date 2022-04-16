#!/usr/bin/env python3

from utils.data import read_sample, write_sample
from sentiment_analysis.sa import sentiment_intensity_analyzer

"""
Script for performing sentiment analysis on a sample
"""


def run_sia():
    """
    Runs sentiment analysis using the VADER algorithm

    Returns
    -------
    None
    """
    data_directory = "data/cleaned_0314"
    results_directory = "results/sia"
    for fname in [
        "putin.csv",
        "russia.csv",
    ]:
        sample = read_sample(f"{data_directory}/{fname}")
        print(f"Performing sentiment analysis on tweets from {fname}")
        positive, negative, neutral = sentiment_intensity_analyzer(sample)
        write_sample(f"{results_directory}/pro_russia.twint", positive)
        write_sample(f"{results_directory}/anti_russia.twint", negative)
        write_sample(f"{results_directory}/neutral.twint", neutral)
    for fname in ["zelensky.csv", "ukraine.csv"]:
        sample = read_sample(f"{data_directory}/{fname}")
        print(f"Performing sentiment analysis on tweets from {fname}")
        positive, negative, neutral = sentiment_intensity_analyzer(sample)
        write_sample(f"{results_directory}/pro_ukraine.twint", positive)
        write_sample(f"{results_directory}/anti_ukraine.twint", negative)
        write_sample(f"{results_directory}/neutral.twint", neutral)
    return None


if __name__ == "__main__":
    run_sia()
