# Part of Speech Tagging

### About

This directory contains various implementations for Part of Speech Tagging, 
from naive to more sophisticated ones.


- #### Approach #1: PoS tagging with Viterbi Algorithm
  This is a naive PoS tagging system using the Viterbi Algorithm.
  It is trained and evaluated on a fraction of  Penn Tree Bank(see below). 
  At first, we split our data into training and testing sets. Then we
  create the transition and emission matrices (see below) in the form of 
  dictionaries. We pass these matrices, along with our test dataset to our
  Viterbi algorithm (see below) implementation. It iterates over the sentences,
  word by word and creates the probability and back-pointer matrices. Using these
  matrices it predicts the Part of Speech each word belongs to in the current
  sentence.


### Reference #1: Penn Tree Bank dataset

Penn Treebank is a well-known dataset, widely used in ML for NLP research.
It contains a large collection of sentences with already tagged words (among other
collections) that we used here. For our purpose, we use the 'tagged_sents()' method
which returns a set of sentences, where each word is tagged.


### Reference #2: Transition and emission matrices

These matrices are often used in NLP for PoS tagging. They are actually the training
phase of our model. Here they are implemented as nested dictionaries.

- Transition matrix: Each row and each column represent all possible PoS tags. Each 
  cell (i, j) contains the probability of encountering the tag indicated by j
  in the next word, if the current word is tagged with the tag indicated by row index i.
  We use the training set,where we already have the correct tag for each word to
  calculate these probabilities for each pair of tags.
- Emission matrix: Each row represents again one of all possible tags. This time, each
  column represents one unique word from our vocabulary (set of all encountered words
  in our training set). Each cell (i, j) contains the probability that, if a word is
  tagged with the tag indicated by row index i, it is actually the word indicated by j:


    P(word=vocabulary[j]|tag=tags[i])
  

### Reference #3: Viterbi Algorithm

The Viterbi algorithm is a well-known algorithm for PoS tagging. It uses the transition
and emission matrices from above to predict the tags for new sentences. For each sentence
it creates the following matrices

- Probability matrix: Each row represents one of all possible tags, while each
  column represents one word from our sentence (in the original order in the sentence). 
  Each cell (i, j) contains the joint probability of the assigned tags for the previous
  words of the sentence (0,..., j-1) and the current word j having the tag indicated
  by row index i. We initialize the first column as the product of the probability of
  going from the start token tag (here 'pi') to the tag indicated by row index i and the
  probability that we encounter the word indicated by j. We get the first from the 
  transition matrix and the latter from the emission matrix.

      transition_matrix['pi'][tags[i]] * emission_matrix[tags[i]][sentence[j]]
  
  For the rest of the columns, for each cell (i, j) we estimate and compare the 
  probabilities of  coming from any cell of the previous column, which means from 
  every tag to the tag indicated by i, which can be written as

      max_k( probability_matrix[k][j - 1] * transition_matrix[tags[k]][tags[i]] * emission_matrix[tags[i]][sentence[j]] )

- Back-pointer matrix: This matrix is used to store the k for which we get the max product 
  from above for each cell (argmax_k). Reaching the end of the sentence and filling both 
  matrices, we can now move backwards using this array to pick the tags that gave us the 
  higher probability in the last column. For the last column, we pick the row i that has
  the highest probability in the probability matrix, which means that the last word will
  be tagged as tags[i]. Then, we move one column to the left and on the row indicated by
  the value of the back-pointer matrix at the current cell. We repeat this process until
  we reach the first column.

In this way, the Viterbi algorithm used Dynamic Programming to pick the sequence of tags
that gives the highest probability for our sentence, using the knowledge stored in the
transition and emission matrices. 

To get a better intuition, [check this video](https://www.youtube.com/watch?v=IqXdjdOgXPM&ab_channel=ritvikmath)

### Instructions

To avoid copyright issues, I didn't upload any corpora on GitHub.
I couldn't get the whole dataset for free legally,
so I used a subset from 
[Kaggle](https://www.kaggle.com/datasets/nltkdata/penn-tree-bank)

All you have to do is download and extract the contents in the script directory.

Requirement: NLTK
