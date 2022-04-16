# Classification of Tweets Related to the 2022 Russian Invasion of Ukraine

This repository contains various data mining methods used for the classification of Tweets 
related to the 2022 Russian Invasion of Ukraine.

This repository contains code written for a report for a class project. The report will be 
included as part of this repository after the report due date.

## Data Directory

* data/cleaned_0314 - Cleaned data used for analysis
* data/labelled - Data labeled as 1 of 7 categories
* data/kmeans-input.twint - 10,000 Tweets with most intense sentiment values. Used as input sample for clustering

## Module Directory

* clustering - Modules used for clustering data
* nn - Modules used for generating a neural network to classify data
* sentiment_analysis - Modules used for performing Sentiment Analysis on data
* utils - Utilities not specific to one method but essential to the project

## Script Directory

* augmented_data_generator.py - Used to generate synthetic data for underrepresented classes using nlp data augmentation methods
* conftest.py - Used for running pytest
* labeller.py - Utility script for labelling tweets
* results.py - Generates data and graphics for representing results from Sentiment Analysis, clustering, and neural network classification
* run_kmeans.py - Runs k-means clustering
* run_nn.py - Runs an Optuna study for determining ideal hyperparameters for a classification neural network
* run_sa.py - Runs Sentiment Analysis
* twint_queries.bash - Script emulating method used for data collection

## Results Directory

* results/sia - Results binned into their expected categories after performing Sentiment Analysis, sorted from most to least intense sentiment. No specific sorting performed on Neutral tweets.
* results/kmeans - Results for each k tested. Includes clustered Tweets as well as most significant features for each cluster. For k=3, files are also included sorting Tweets by their distance to the cluster centroid.
* results/nn - Results of Optuna trials for each class size tested

### MLflow Results

Results for the neural network can also be viewed via the MLflow UI. This can be done by:

* Install MLflow via `pip install mlflow`
* Run `mlflow ui`
* Navigate to http://localhost:5000
