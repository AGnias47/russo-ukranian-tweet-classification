#!/usr/bin/env python3

"""
Script for performing sentiment analysis on a sample
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tqdm


def sentiment_intensity_analyzer(sample):
    """
    Separates sample into three separate lists of positive, negative, and neutral

    Parameters
    ----------
    sample (list): list of twint dicts

    Returns
    -------
    list, list, list
        positive, negative, and neutral entries, respectively
    """
    sid = SentimentIntensityAnalyzer()
    pos, neg, neu = list(), list(), list()
    for tweet in tqdm.tqdm(sample):
        compound_score = sid.polarity_scores(tweet.raw_content)["compound"]
        tweet.sentiment = compound_score
        if compound_score > 0:
            pos.append(tweet)
        elif compound_score < 0:
            neg.append(tweet)
        else:
            neu.append(tweet)
    pos = sorted(pos, key=lambda x: x.sentiment, reverse=True)
    neg = sorted(neg, key=lambda x: x.sentiment, reverse=False)
    return pos, neg, neu
