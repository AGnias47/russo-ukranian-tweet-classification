#!/usr/bin/env python3

"""
Defines the Objective Function used for Optuna hyperparameter tuning
"""

from nn.nn import NeuralNet, train, evaluate

import optuna
import torch
import tqdm


class OptunaObjective:
    """
    Objective function used by Optuna
    """

    def __init__(self, train_dataloader, test_dataloader, validation_dataloader, epochs, vocab_size, classes, device):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.valid_dataloader = validation_dataloader
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.classes = classes
        self.device = device

    def __call__(self, trial):
        """
        Trains and tests the model through an Optuna study. Variable optimizer, learning rate, and gamma values are used
        in order to determine most effective values.

        Meant to be run via an Optuna study

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        float
            Accuracy value of the model
        """
        input_layer = trial.suggest_int("input_layer", 16, 32)
        output_layer = trial.suggest_int("output_layer", self.classes, 25)
        model = NeuralNet(self.vocab_size, input_layer, output_layer).to(self.device)
        loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 5e-5, 5e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
        # Train the model
        total_accuracy = 0
        for epoch in tqdm.tqdm(range(self.epochs)):
            train(model, self.train_dataloader, optimizer, loss_function)
            current_accuracy = evaluate(model, self.valid_dataloader)
            trial.report(current_accuracy, epoch)
            if total_accuracy > current_accuracy:
                scheduler.step()
            else:
                total_accuracy = current_accuracy
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned(f"Pruned after epoch {epoch}")
        # Test the model
        accu_test = evaluate(model, self.test_dataloader)
        print(f"Test Accuracy: {round(accu_test, 3)}")
        return accu_test
