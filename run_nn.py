#!/usr/bin/env python3

from nn.data import generate_dataloaders_with_vocab
from nn.objective import OptunaObjective
from nn.nn import NeuralNet, train, evaluate

import optuna
import torch
import torch.utils.data
import tqdm
from optuna.integration.mlflow import MLflowCallback

"""
Trains and tests the data on the neural network. Uses optuna hyperparameter optimization library to determine best 
hyperparameters and optimizer algorithm.
"""

EPOCHS = 25
STUDY_LENGTH_SECONDS = 600
CLASSES = 3


def run_known_model(
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    vocab_length,
    input_layer,
    output_layer,
    lr,
    gamma,
    optim,
    optuna_accuracy=None,
):
    """
    Trains and tests the neural network with set parameters. If optuna_accuracy is provided, the accuracy of the model
    used in this function will be compared to the accuracy of the Optuna study.

    Parameters
    ----------
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    test_dataloader: DataLoader
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
    optuna_accuracy (default is None): float
        If provided, compares the best accuracy from the Optuna study to the accuracy calculated from the model used
        in this function

    Returns
    -------
    float
        Accuracy of model
    """
    model = NeuralNet(vocab_length, input_layer, output_layer).to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
    # Train the model
    total_accuracy = 0
    for _ in tqdm.tqdm(range(EPOCHS)):
        train(model, train_dataloader, optimizer, loss_function)
        current_accuracy = evaluate(model, valid_dataloader)
        if total_accuracy > current_accuracy:
            scheduler.step()
        else:
            total_accuracy = current_accuracy
    # Test the model
    accu_test = evaluate(model, test_dataloader)
    if optuna_accuracy:
        if accu_test > optuna_accuracy:
            print(
                f"Known model of ({input_layer}, {output_layer}, {lr}, {gamma}, {optim.__name__})"
                + f" performed better than Optuna trials with an accuracy of {accu_test}"
            )
        elif accu_test == optuna_accuracy:
            print(
                f"Known model of ({input_layer}, {output_layer}, {lr}, {gamma}, {optim.__name__})"
                + f" performed equal to Optuna trials; each had an accuracy of {accu_test}"
            )
        else:

            print(f"Optuna exceeded known model's performance ({accu_test})")
    return accu_test


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader, valid_dataloader, vocab = generate_dataloaders_with_vocab(
        device, classes=CLASSES
    )
    sample_len = len(train_dataloader.dataset) + len(test_dataloader.dataset) + len(valid_dataloader.dataset)
    study = optuna.create_study(
        study_name=f"NN Optimization: {sample_len} Samples, {CLASSES} Classes, {device.type}, augmented data",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    try:
        study.optimize(
            OptunaObjective(train_dataloader, test_dataloader, valid_dataloader, EPOCHS, len(vocab), CLASSES, device),
            timeout=STUDY_LENGTH_SECONDS,
            callbacks=[MLflowCallback(metric_name="accuracy")],
        )
    except KeyboardInterrupt:
        pass
    print("Optuna study best trial:")
    trial = study.best_trial
    accuracy = trial.value
    print(f"Value: {accuracy}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
