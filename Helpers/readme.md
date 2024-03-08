# Helpers

### About

This directory contains various models, algorithms and tools that can be used
in several tasks as building blocks.


- #### Helper #1: CBoW
  CBoW (Continuous Bag of Words) is a method for representing words as embeddings,
  which are vectors of numbers. In an NLP task, one can choose some properties to
  describe words of the vocabulary, such as the genre or if the word encloses a
  concrete or abstract meaning (e.g. stone vs idea). In CBoW, we don't
  specify what these features are, but we let the model learn and decide the
  ones that lead to similar embeddings for words with similar meaning. The process
  begins by representing each word as a one-hot vector. We specify a window size
  and use it in the sentences of the training dataset. As we slide the window,
  the middle word (context word) is the target word that the model tries to predict
  given the words that surround it (window_size//2 words to the left and to the
  right of the context word). Eventually, the model learns the word embeddings, 
  even though that was not its goal. We can then extract these embeddings and 
  use them for several tasks like sentiment analysis, autocomplete, analogies etc.