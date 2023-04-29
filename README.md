# Classification of Tweets Related to the 2022 Russian Invasion of Ukraine

This repository contains various data mining methods used for the classification of Tweets related to the 2022 Russian Invasion of Ukraine. This was done for a class project where the goal was to create and analyze methods for automatically determining whether a Tweet was expressing support for or against a certain side of the conflict. The full report can be found [here](final-report/report.pdf).

Tweet data was obtained using Twint, a Twitter data mining tool, to pull Tweets relevant to the conflict. Sentiment Analysis, k-means clustering, and a Neural Network were used to label these Tweets, and results for each method were compared.

## Summary

Tweets were downloaded using the Twitter Intelligence Tool (Twint) by querying for hashtags relevant to the Russo-ukrainian war andwere used in the following methods:

* Sentiment Analysis - Performed using a VADER SA model via `nltk`. Hypothesis was that by querying Tweets with a certain hashtag, the sentiment score would indicate whether the Tweet was for or against a certain side of the war
* Clustering - k-means clustering with PCA decomposition. Hypothesis was that Tweets for or against a certain side of the war would be clustered together
* Classification Neural Network - created with PyTorch. Hypothesis was that given labeled Tweets, the neural network could determine the category of unlabeled Tweets

Overall, the methods showed some interesting correlations, but were not effective enough on their own to accurately determine support or opposition for a certain side of the war. They did show effectiveness in grouping similar Tweets, and would probably be most useful as a preprocessing step in a more complex model, or for assistance with manually labelling Tweets.

## Data Directory

* [data/cleaned_0314](data/cleaned_0314) - Cleaned data used for analysis
* [data/labelled](data/labelled) - Data labeled as 1 of 7 categories
* [data/kmeans-input.twint](data/kmeans-input.twint) - 10,000 Tweets with most intense sentiment values. Used as input sample for clustering

## Script Directory

### Data

* `twint_queries.bash` - Script emulating method used for data collection
* `augmented_data_generator.py` - Used to generate synthetic data for underrepresented classes using nlp data augmentation methods
* `labeller.py` - Utility script for labelling tweets

### Methods

* `run_kmeans.py` - Runs k-means clustering
* `run_nn.py` - Runs an Optuna study for determining ideal hyperparameters for a classification neural network
* `run_sa.py` - Runs Sentiment Analysis

### Results

* `results.py` - Generates data and graphics for representing results from Sentiment Analysis, clustering, and neural network classification

## Results Directory

* [results/sia](results/sia) - Results binned into their expected categories after performing Sentiment Analysis, sorted from most to least intense sentiment. No specific sorting performed on Neutral tweets.
* [results/kmeans](results/kmeans) - Results for each k tested. Includes clustered Tweets as well as most significant features for each cluster. For k=3, files are also included sorting Tweets by their distance to the cluster centroid.
* [results/nn](results/nn) - Results of Optuna trials for each class size tested

### MLflow Results

Results for the neural network can also be viewed via the MLflow UI. This can be done by:

* Install MLflow via `pip install mlflow`
* Run `mlflow ui`
* Navigate to http://localhost:5000
