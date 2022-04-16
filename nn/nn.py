#!/usr/bin/env python3

"""
Neural network for text processing.

Adapted from:
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
"""

import torch
from torch import nn


class NeuralNet(nn.Module):
    """
    Neural Network used for text classification
    """

    def __init__(self, vocab_size, embed_dim, num_class):
        """
        Initialize neural network

        Parameters
        ----------
        vocab_size: int
            Size of vocab in corpus of data being classified
        embed_dim: int
            Size of input layer
        num_class: int
            Size of output layer
        """
        super(NeuralNet, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights

        Returns
        -------
        None
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        return None

    def forward(self, text, offsets):
        """
        Forward function for neural network

        Parameters
        ----------
        text: Collection
        offsets: Collection

        Returns
        -------
        Step result
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def train(model, dataloader, optimizer, loss_function):
    """
    Train the model

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
        Training data
    optimizer: torch.optim.*
    loss_function: torch.nn.CrossEntropyLoss
        Or equivalent

    Returns
    -------
    None
    """
    model.train()
    for label, text, offset in dataloader:
        optimizer.zero_grad()
        predicted_label = model(text, offset)
        loss = loss_function(predicted_label, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    return None


def evaluate(model, dataloader):
    """
    Evaluate the model

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
        Test / Validation data

    Returns
    -------
    float
        Accuracy
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (label, text, offset) in dataloader:
            predicted_label = model(text, offset)
            correct += (predicted_label.argmax(1) == label).sum().item()
            total += label.size(0)
    return correct / total


def evaluate_confusion_matrix(model, dataloader):
    """
    Evaluation function returning data necessary for confusion matrix

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
        Test data

    Returns
    -------
    Collection, Collection, float
        Predictions, Actual labels, accuracy
    """
    model.eval()
    correct, total = 0, 0
    y_pred, y_true = list(), list()
    with torch.no_grad():
        for (label, text, offset) in dataloader:
            predicted_label = model(text, offset)
            y_pred.extend(predicted_label.argmax(1))
            y_true.extend(label)
            correct += (predicted_label.argmax(1) == label).sum().item()
            total += label.size(0)
    return y_pred, y_true, correct / total
