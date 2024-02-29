# Autocomplete

### About

This directory contains various implementations for Autocomplete, 
from naive to more sophisticated ones.


- #### Approach #1: n-gram model
  This is a naive autocorrect system that uses n-grams to define the most
  probable word to follow a given sequence. At first, we split the data into
  train and test set, convert all words to lowercase, tokenize the sentences 
  and create a vocabulary that consists of words that appear at least twice in
  the training set. Then we substitute all words that are not included in the 
  vocabulary with the unknown word token '<unk'.
  After preprocessing the data, we count the occurrences of all sequences that 
  appear in the training set that have a particular length. For example, in case
  we wish to decide on the next word using the two previous words (bigram), we
  count all bigrams and trigrams in the training set. These counts will give us
  the probability of a word following the given sequence.
  Also, we can simulate the process of starting to type the next word and the
  autocomplete system will give us suggestions that start with the typed 
  characters.

### Instructions

To avoid copyright issues, I didn't upload any corpora on GitHub.
I used the data from the 'en_US.twitter.txt' file in
[this GitHub repository](https://github.com/amanjeetsahu/Natural-Language-Processing-Specialization/tree/master/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%203)

All you have to do is download the file and move it in the script directory.

Requirements: NLTK, sklearn
