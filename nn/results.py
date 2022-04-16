#!/usr/bin/env python3

"""
Train Neural Networks with established hyperparameters and process results
"""

from .nn import NeuralNet, train, evaluate, evaluate_confusion_matrix
from .data import (
    generate_dataloaders_with_vocab,
    Collator,
    yield_tokens,
    filter_sample_and_update_classes,
    filter_sample_and_update_classes_with_augmented_data,
)
from utils.labels import (
    INT_TO_LONGFORM_DICT,
    label_to_int,
    generate_class_labels,
)
from utils.utils import seed_everything

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.utils.data
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

import tqdm

EPOCHS = 25
K_FOLD = StratifiedKFold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)


def generate_trained_model(
    train_dataloader, valid_dataloader, vocab_length, input_layer, output_layer, lr, gamma, optim
):
    """

    Parameters
    ----------
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    vocab_length: int
        Length of the vocab of texts used
    input_layer: int
        Input layer size of the neural network
    output_layer: int
        Output layer size of the neural network
    lr: float
        Learning rate value
    gamma: float
        Gamma value
    optim: torch.optim.*
        Optimizer function

    Returns
    -------
    Collection, Collection
    """
    model = NeuralNet(vocab_length, input_layer, output_layer).to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
    # Train the model
    total_accuracy = 0
    for _ in tqdm.tqdm(range(EPOCHS)):
        train(model, train_dataloader, optimizer, loss_function)
        if valid_dataloader:
            current_accuracy = evaluate(model, valid_dataloader)
            if total_accuracy > current_accuracy:
                scheduler.step()
            else:
                total_accuracy = current_accuracy
        else:
            scheduler.step()
    return model


def run_via_holdout(sample, input_layer, output_layer, lr, gamma, optim, classes=3):
    """
    Re-trains and tests a Neural Network based on given parameters using the holdout method, and plots results as a
    confusion matrix.

    Parameters
    ----------
    sample: Collection
    input_layer: int
    output_layer: int
    lr: float
    gamma: float
    optim: torch.optim.*
    classes: int (default is 3)

    Returns
    -------
    None
    """
    train_dataloader, test_dataloader, valid_dataloader, vocab = generate_dataloaders_with_vocab(
        device, sample, classes
    )
    if classes == 3:
        class_labels = [INT_TO_LONGFORM_DICT[1], INT_TO_LONGFORM_DICT[2], INT_TO_LONGFORM_DICT[5]]
    else:
        class_labels = [INT_TO_LONGFORM_DICT[i + 1] for i in range(classes)]
    model = generate_trained_model(
        train_dataloader, valid_dataloader, len(vocab), input_layer, output_layer, lr, gamma, optim
    )
    y_pred, y_real, accuracy = evaluate_confusion_matrix(model, test_dataloader)
    cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_real)
    df_cm = pd.DataFrame(cf_matrix, index=class_labels, columns=class_labels)
    plt.figure(figsize=(12, 7))
    cf = sn.heatmap(df_cm, annot=True, cmap="flare")
    cf.set_xlabel(f"Predicted - {round(accuracy*100, 2)}% Accuracy", fontsize=12)
    cf.set_ylabel("Actual", fontsize=12)
    plt.savefig(f"final-report/images/nn-{classes}.png")
    plt.show()
    return None


def run_via_cross_validation(sample, classes, input_layer, output_layer, lr, gamma, optim, folds=5):
    """
    Re-trains and tests a Neural Network based on given parameters using k-fold cross-validation, and plots results as a
    confusion matrix.

    Parameters
    ----------
    sample: Collection
    classes: int
    input_layer: int
    output_layer: int
    lr: float
    gamma: float
    optim: torch.optim.*
    folds: int (default is 5)

    Returns
    -------
    Collection
        Individual accuracies for each k-fold
    """
    d = list()
    orig_len = len(sample)
    sample = filter_sample_and_update_classes(sample, classes)
    for tweet in sample:
        d.append((tweet.raw_content, label_to_int(tweet.label)))
    sample_len = len(sample)
    labels = [label_to_int(x.label) for x in sample]
    label_count = len(set(labels))
    print(f"Using {sample_len} tweets with {label_count} different classes (filtered {orig_len - sample_len} tweets)")
    vocab = build_vocab_from_iterator(yield_tokens(sample), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    collator = Collator(vocab, device)
    d_map = to_map_style_dataset(d)
    kfold = K_FOLD(n_splits=folds, shuffle=True, random_state=42)
    y_preds, y_reals, accuracies = list(), list(), list()
    for train_index, test_index in kfold.split(sample, labels):
        trainset = torch.utils.data.SubsetRandomSampler(train_index)
        testset = torch.utils.data.SubsetRandomSampler(test_index)
        trainloader = torch.utils.data.DataLoader(d_map, batch_size=8, sampler=trainset, collate_fn=collator)
        testloader = torch.utils.data.DataLoader(d_map, batch_size=8, sampler=testset, collate_fn=collator)
        model = generate_trained_model(trainloader, None, len(vocab), input_layer, output_layer, lr, gamma, optim)
        y_pred, y_real, accuracy = evaluate_confusion_matrix(model, testloader)
        y_preds += y_pred
        y_reals += y_real
        accuracies.append(accuracy)
    # Build confusion matrix
    generate_class_labels(classes)
    cf_matrix = confusion_matrix(y_pred=y_preds, y_true=y_reals)
    class_labels = generate_class_labels(classes)
    df_cm = pd.DataFrame(cf_matrix[0:classes, 0:classes], index=class_labels, columns=class_labels)
    plt.figure(figsize=(12, 7))
    cf = sn.heatmap(df_cm, annot=True, cmap="flare")
    f1 = f1_score(y_true=y_reals, y_pred=y_preds, average="weighted")
    cf.set_xlabel(
        f"Predicted - {round(sum(accuracies) / folds *100, 2)}% Mean Accuracy of Range "
        + f"{round(min(accuracies)*100, 2)}-{round(max(accuracies)*100,2)}% "
        + f"with a weighted F1 score of {round(f1, 2)} on {folds} folds",
        fontsize=12,
    )
    cf.set_ylabel("Actual", fontsize=12)
    plt.savefig(f"final-report/images/nn-cv-{classes}.png")
    plt.show()
    get_individual_class_accuracy(cf_matrix)
    return accuracies


def run_via_cross_validation_augmented(sample, classes, input_layer, output_layer, lr, gamma, optim, folds=5):
    """
    Re-trains and tests a Neural Network based on given parameters using k-fold cross-validation, and plots results as a
    confusion matrix. Includes augmented data in training data.

    Parameters
    ----------
    sample: Collection
    classes: int
    input_layer: int
    output_layer: int
    lr: float
    gamma: float
    optim: torch.optim.*
    folds: int (default is 5)

    Returns
    -------
    Collection
        Individual accuracies for each k-fold
    """
    orig_len = len(sample)
    sample, start_aug, end_aug = filter_sample_and_update_classes_with_augmented_data(sample, classes)
    augmented_count = end_aug - start_aug
    d = list()
    for tweet in sample:
        d.append((tweet.raw_content, label_to_int(tweet.label)))
    labels = [label_to_int(x.label) for x in sample]
    label_count = len(set(labels))
    print(
        f"Using {end_aug} tweets with {label_count} different classes (filtered {orig_len - start_aug} "
        + f"tweets and added {augmented_count} tweets from augmented data)"
    )
    vocab = build_vocab_from_iterator(yield_tokens(sample), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    collator = Collator(vocab, device)
    d_map = to_map_style_dataset(d)
    augmented_fold = int((end_aug - start_aug) / 5)
    kfold = K_FOLD(n_splits=folds, shuffle=True, random_state=42)
    y_preds, y_reals, accuracies = list(), list(), list()
    i = 0
    for train_index, test_index in kfold.split(sample[0:start_aug], labels[0:start_aug]):
        train_index = np.append(
            train_index, np.arange(start_aug + augmented_fold * i, start_aug + augmented_fold * (i + 1), dtype=int)
        )
        trainset = torch.utils.data.SubsetRandomSampler(train_index)
        testset = torch.utils.data.SubsetRandomSampler(test_index)
        trainloader = torch.utils.data.DataLoader(d_map, batch_size=8, sampler=trainset, collate_fn=collator)
        testloader = torch.utils.data.DataLoader(d_map, batch_size=8, sampler=testset, collate_fn=collator)
        model = generate_trained_model(trainloader, None, len(vocab), input_layer, output_layer, lr, gamma, optim)
        y_pred, y_real, accuracy = evaluate_confusion_matrix(model, testloader)
        y_preds += y_pred
        y_reals += y_real
        accuracies.append(accuracy)
        i += 1
    # Build confusion matrix
    class_labels = generate_class_labels(classes)
    cf_matrix = confusion_matrix(y_pred=y_preds, y_true=y_reals)
    df_cm = pd.DataFrame(cf_matrix[0:classes, 0:classes], index=class_labels, columns=class_labels)
    plt.figure(figsize=(12, 7))
    cf = sn.heatmap(df_cm, annot=True, cmap="flare")
    f1 = f1_score(y_true=y_reals, y_pred=y_preds, average="weighted")
    cf.set_xlabel(
        f"Predicted - {round(sum(accuracies) / folds *100, 2)}% Mean Accuracy of Range "
        + f"{round(min(accuracies)*100, 2)}-{round(max(accuracies)*100,2)}% "
        + f"with a weighted F1 score of {round(f1, 2)} on {folds} folds",
        fontsize=12,
    )
    cf.set_ylabel("Actual", fontsize=12)
    plt.savefig(f"final-report/images/nn-cv-{classes}-augmented.png")
    plt.show()
    get_individual_class_accuracy(cf_matrix)
    return accuracies


def get_individual_class_accuracy(cf):
    """
    Get accuracy for individual classes from Confusion Matrix and prints accuracies as a LaTeX table

    Parameters
    ----------
    cf: ndarray

    Returns
    -------
    DataFrame
    """
    accuracy = cf.diagonal() / cf.sum(axis=1)
    df = pd.DataFrame(
        [
            {
                "Pro-Ukraine": f"{round(accuracy[0] * 100, 2)}%",
                "Pro-Russia": f"{round(accuracy[1] * 100, 2)}%",
                "Neutral": f"{round(accuracy[2] * 100, 2)}%",
            }
        ]
    )
    print(df.to_latex())
    return df
