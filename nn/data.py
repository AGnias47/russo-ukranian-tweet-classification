#!/usr/bin/env python3

"""
Used to prepare data for neural network
"""

from utils.data import read_sample
from utils.labels import label_to_int

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, ConcatDataset
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator


torch.manual_seed(42)
# Hyperparameters
BATCH_SIZE = 8
NGRAMS = 2
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def yield_tokens(sample):
    """
    Tokenizes the raw contents and performs n-gram tokenization

    Parameters
    ----------
    sample: list

    Returns
    -------

    """
    tokenizer = get_tokenizer("basic_english")
    for tweet in sample:
        yield ngrams_iterator(tokenizer(tweet.raw_content), NGRAMS)


class Collator:
    """
    Class for collating batch data
    """

    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device

    def __call__(self, batch):
        tokenizer = get_tokenizer("basic_english")
        labels, tweets_list, offsets = list(), list(), [0]
        for (tweet, label) in batch:
            labels.append(label - 1)
            processed_text = torch.tensor(
                self.vocab(list(ngrams_iterator(tokenizer(tweet), NGRAMS))), dtype=torch.int64
            )
            tweets_list.append(processed_text)
            offsets.append(processed_text.size(0))
        labels = torch.tensor(labels, dtype=torch.int64)
        tweets_list = torch.cat(tweets_list)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        return labels.to(self.device), tweets_list.to(self.device), offsets.to(self.device)


def load_data(paths):
    """
    Loads data from files into a list of dicts

    Parameters
    ----------
    paths: Collection

    Returns
    -------
    Collection
        List of tweets stored as dicts
    """
    sample = list()
    for p in paths:
        if Path(p).exists():
            for fpath in Path(p).glob("*.twint"):
                sample += read_sample(str(fpath), label=True)
        else:
            raise ValueError(f"Invalid path {Path.cwd()}/{p} specified")
    return sample


def generate_dataloaders_with_vocab(device, sample=None, classes=5):
    """
    Generates train, test, and validation DataLoaders from a sample as well as returns the vocab length

    Parameters
    ----------
    device: torch.device
    sample: Collection
    classes: int
        Number of potential labels. Only handles 2, 5, and 7

    Returns
    -------
    DataLoader, DataLoader, DataLoader, list
    """
    if not sample:
        sample = load_data(["data/labelled/sia", "data/labelled/cluster"])
    orig_len = len(sample)
    sample, start_aug, end_aug = filter_sample_and_update_classes_with_augmented_data(sample, classes)
    augmented_count = end_aug - start_aug
    main_sample, augmented_sample = list(), list()
    for i, tweet in enumerate(sample):
        if i < start_aug:
            main_sample.append((tweet.raw_content, label_to_int(tweet.label)))
        else:
            augmented_sample.append((tweet.raw_content, label_to_int(tweet.label)))
    label_count = len(set(x.label for x in sample))
    print(
        f"Using {end_aug} tweets with {label_count} different classes (filtered {orig_len - start_aug} "
        + f"tweets and added {augmented_count} tweets from augmented data)"
    )
    vocab = build_vocab_from_iterator(yield_tokens(sample), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    collator = Collator(vocab, device)
    d_map_main = to_map_style_dataset(main_sample)
    d_map_augmented = to_map_style_dataset(augmented_sample)
    training_instances = int(len(d_map_main) * TRAIN_SPLIT)
    train_set, test_val_set = random_split(d_map_main, [training_instances, len(d_map_main) - training_instances])
    train_set = ConcatDataset([train_set, d_map_augmented])
    # Subset(dataset, indices[offset - length: offset])
    test_instances = int(len(test_val_set) * VAL_SPLIT)
    test_set, val_set = random_split(test_val_set, [test_instances, len(test_val_set) - test_instances])
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator),
        vocab,
    )


def filter_sample_and_update_classes(sample, classes):
    """
    Filters the sample and updates the class labels based on the number of classes being used in the analysis

    Parameters
    ----------
    sample: Collection
    classes: int

    Returns
    -------
    Collection
        Filtered sample
    """
    if classes == 2:
        sample = list(filter(lambda x: x.label not in {"u", "us", "n"}, sample))
        for s in sample:
            if s.label in {"pu", "ar"}:
                s.label = "pu"
            elif s.label in {"pr", "au"}:
                s.label = "pr"
            else:
                raise ValueError
    if classes == 3:
        sample = list(filter(lambda x: x.label not in {"u", "us"}, sample))
        for s in sample:
            if s.label in {"pu", "ar"}:
                s.label = "pu"
            elif s.label in {"pr", "au"}:
                s.label = "pr"
            elif s.label == "n":
                s.label = "au"
            else:
                raise ValueError
    elif classes == 5:
        sample = list(filter(lambda x: x.label not in {"u", "us"}, sample))
    else:
        pass
    return sample


def filter_sample_and_update_classes_with_augmented_data(sample, classes):
    """
    Filters the sample and updates the class labels based on the number of classes being used in the analysis, as well
    as adds augmented data to the end of the sample

    Parameters
    ----------
    sample: Collection
    classes: int

    Returns
    -------
    Collection, int, int
        Filtered sample, start index for augmented data, end index for augmented data
    """
    if classes == 2:
        sample = list(filter(lambda x: x.label not in {"u", "us", "n"}, sample))
        start_aug = len(sample)
        augmented_data = load_data(["data/labelled/augmented"])
        sample += augmented_data
        end_aug = len(sample)
        for s in sample:
            if s.label in {"pu", "ar"}:
                s.label = "pu"
            elif s.label in {"pr", "au"}:
                s.label = "pr"
            else:
                raise ValueError
    elif classes == 3:
        sample = list(filter(lambda x: x.label not in {"u", "us"}, sample))
        start_aug = len(sample)
        augmented_data = load_data(["data/labelled/augmented"])
        sample += augmented_data
        end_aug = len(sample)
        for s in sample:
            if s.label in {"pu", "ar"}:
                s.label = "pu"
            elif s.label in {"pr", "au"}:
                s.label = "pr"
            elif s.label == "n":
                s.label = "au"
            else:
                raise ValueError
    elif classes == 5:
        sample = list(filter(lambda x: x.label not in {"u", "us"}, sample))
        start_aug = len(sample)
        augmented_data = load_data(["data/labelled/augmented"])
        sample += augmented_data
        end_aug = len(sample)
    else:
        start_aug = len(sample)
        augmented_data = load_data(["data/labelled/augmented"])
        sample += augmented_data
        end_aug = len(sample)
    return sample, start_aug, end_aug
