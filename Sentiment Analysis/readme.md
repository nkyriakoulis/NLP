# Sentiment Analysis

### About

This directory contains various implementations for Sentiment Analysis, 
from naive to more sophisticated ones.


- #### Approach #1: Naive sentiment analysis
  This is a naive sentiment analysis system using a Naive Bayes Classifier.
  It is trained and evaluated on several movie reviews (see below). There
  are two versions, one with and one without Stemming. At first, we iterate
  through the training set and preprocess our data: We split into words,
  convert to lowercase, remove punctuation, special characters and
  "Stop words" and perform "Stemming" (see below). Then, we count the
  frequency of each word in positive and negative review. When finished,
  we apply Laplacian Smoothing and convert these frequencies to Probabilities.
  For each word we compute the log likelihood for


    P(word|positive review) / P(word|negative review)
  
  
  This concludes the training phase of our model. Continuing with the evaluation
  phase, we iterate over our test dataset, repeat the preprocessing steps and then
  sum um the log likelihoods for all the words in our corpus. The model classifies
  a review as "Positive", "Neutral" or "Negative" if the sum is greater than, 
  equal to or smaller than zero.
  It is "Naive", because it doesn't take into consideration factors like the context
  in which a word appears, punctuation and semantics. It also doesn't remove naming
  entities and other meaningless for our purpose words (e.g. URLs).


### Reference: NLTK Library

Here I use NLTK (Natural Language Toolkit) for

- Removing "Stop Words": Stop Words are words that are frequently encountered in
  speech but do not offer any value in our task (e.g 'the', 'an', 'in', 'under' etc.).
  NLTK offers a list of such words.
- Stemming: Stemming means reducing words to their root form, by trimming
  (e.g. 'doing' -> 'do', 'happiness' -> 'happi' etc.). This helps us "group" words
  that come from the same root and consider them same in our vocabulary (e.g. 'happy'
  and 'happiness'), as they have similar meaning.

More about NLTK can be found 
[here](https://www.nltk.org/).

### Instructions

To avoid copyright issues, I didn't upload any corpora on GitHub. 
To run the script, you can download the dataset I also used 
[here](https://ai.stanford.edu/~amaas/data/sentiment/).
