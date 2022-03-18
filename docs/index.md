This is the documentation for the challenge, how it works and what I did during this week.

<center>
<img src="img/pizza-i-love-pizza.gif">

<em>Me working on the challenge...</em>
</center>

## Planning

I started by thinking about the problem in a way that I could understand it.

**What is the challenge goal?** 

**What is the problem?**

**What type of data do we have?** 

**What do I need to build as a solution?**

This is the plan I have set out for the challenge:

![challenge plan](./img/planning.jpeg)

I followed most of the plan and its order, but machine learning problems are more and iterative process than a linear one.

Since the beginning of the challenge I wanted to try a classic machine learning model to solve the problem, and then
try a deep learning model to see which one is better and if it's worth it to use for the challenge.

## 🧹 Cleaning and Exploring the Data - [Notebook](https://github.com/ChainYo/pizza-challenge/blob/master/notebooks/cleaning.ipynb)

I started the exploration with a bit of data cleaning.

There was some missing data in the dataset and some unnecessary data.

## 🔎 More Data Exploration - [Notebook](https://github.com/ChainYo/pizza-challenge/blob/master/notebooks/exploration.ipynb)

Then, I started really exploring the data.

I focused on text data which is the most relevant for this challenge in my opinion.

So, I cleaned with `regex` request text data and did some wordclouds.

Here is the wordcloud of the most common words in the `requester_text` column:

![wordcloud](./img/wordcloud.png)

I also found that negative requests contain a lot of misspellings.

## ⚙️ Preprocessing - [Notebook](https://github.com/ChainYo/pizza-challenge/blob/master/notebooks/preprocessing.ipynb)

Then I started to preprocess the data before building any model.

I have replaced `True` and `False` with `1` and `0` in the `requester_received_pizza` column which will be the target column.

I also arranged some columns in a way that I could use them for a training if needed.

## 🧭 First Model - [Notebook](https://github.com/ChainYo/pizza-challenge/blob/master/notebooks/random_forest_classifier.ipynb)

I started with the idea to build a simple `RandomForestClassifier` model on only the text data.

I used an old notebook I had written months ago where I used `nltk` and `gensim.word2vec` to build a word2vec model.

So I used `nltk` to tokenize the text data in order to build the vocabulary of the model. This model can be used 
to create clusters of words that are similar to each other. 

226 clusters were created and used to create a mapping between the words and their clusters. 

This mapping is used to create multiple bags of centroids, which are used to create the features for the model.

With the dataset created, I started to train the model. I used `Optuna` to optimize the hyperparameters of the model and
to help me find the best model.

![best classifier](./img/classifier-conf-matrix.png)

As you can see, the model is not very good. 😅

Because, I'm not really confident in the model choice and the tools I used, I decided to give up on this approach and
try a deep learning model, which passion me more.

## 🧠 Deep Learning Model - [Notebook](https://github.com/ChainYo/pizza-challenge/blob/master/notebooks/deep_learning.ipynb) | [Implementation](https://github.com/ChainYo/pizza-challenge/blob/master/src/pizza_challenge/pipelines/training/model.py)

I switched to a deep learning model and started to think about the problem and which model to use.

I decided to fine tune a transformers model on a downstream task, which is a classification task in this case.

As a backbone of the model, I used a `bert-base-uncased` model which is a pretrained model from the huggingface library.

The model has been trained to be able to read and understand english text, so it will be fine for our task.

In order to fine tune the model, I have added a simple classification layer to the model.

I used `PyTorch-Lightning` for the implementation of the model, because it's the best library for implementing deep
learning models and deploying them on a production environment.



