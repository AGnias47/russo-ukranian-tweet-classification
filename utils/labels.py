#!/usr/bin/env python3

"""
Utilities for handling data labels
"""

LABEL_DICT = {1: "pu", 2: "pr", 3: "au", 4: "ar", 5: "n", 6: "us", 7: "u"}
LONGFORM_LABEL_DICT = {
    "pu": "pro-ukraine",
    "pr": "pro-russia",
    "au": "anti-ukraine",
    "ar": "anti-russia",
    "n": "neutral",
    "us": "other-entity",
    "u": "unknown",
}
INT_TO_LONGFORM_DICT = {
    1: "pro-ukraine",
    2: "pro-russia",
    3: "anti-ukraine",
    4: "anti-russia",
    5: "neutral",
    6: "other-entity",
    7: "unknown",
}
color_dict = {
    "pro-ukraine": "#2378d1",
    "pro-russia": "#d53b30",
    "anti-ukraine": "#2CA02C",
    "anti-russia": "#ff7f0e",
    "neutral": "#ffc300",
    "other-entity": "#9467BD",
    "unknown": "grey",
}


def int_to_label(n):
    """
    Converts an int to its corresponding class label

    Parameters
    ----------
    n: int

    Returns
    -------
    str
        Class label corresponding to int
    """
    return LABEL_DICT[n]


def label_to_int(label):
    """
    Converts a label to its corresponding int

    Parameters
    ----------
    label: str

    Returns
    -------
    int
        Int corresponding to class label
    """
    return {v: k for k, v in LABEL_DICT.items()}[label]


def label_longform(s):
    """
    Return longform of labels; used for presenting results

    Parameters
    ----------
    s: str
        Shortform string label

    Returns
    -------
    str
        Longform string label
    """
    return LONGFORM_LABEL_DICT[s]


def assign_biclass_labels(sample):
    """
    Converts a short label into a verbose one; used for presenting results

    Parameters
    ----------
    sample: Tweet

    Returns
    -------
    None
    """
    if sample.label == "anti-russia":
        sample.label = "pro-ukraine"
    if sample.label == "anti-ukraine":
        sample.label = "pro-russia"
    if sample.label in {"other-entity", "unknown"}:
        sample.label = "neutral"
    return None


def generate_class_labels(classes):
    """
    Generates a list of class labels based on the number of classes in the analysis

    Parameters
    ----------
    classes: int

    Returns
    -------
    Collection
        Classes in expected order
    """
    if classes == 3:
        return [INT_TO_LONGFORM_DICT[1], INT_TO_LONGFORM_DICT[2], INT_TO_LONGFORM_DICT[5]]
    else:
        return [INT_TO_LONGFORM_DICT[i + 1] for i in range(classes)]
