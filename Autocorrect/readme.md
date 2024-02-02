# Naive Autocorrect Project in Python

### About

This directory contains various implementations for Autocorrect Systems, 
from naive to more sophisticated ones.


- #### Approach #1: Naive autocorrect
  This is a naive autocorrect method using minimum edit distances. 
  At first, it creates a dictionary to be used as our vocabulary, 
  storing all the words and the frequency they are encountered
  in the corpus. Then, we provide a list of words, possibly 
  misspelled and ask the script to give us up to 3 most 
  probable suggestions (if they exist), in case of a misspelling.
  
  The method is "Naive" because it doesn't take the context of
  the word into consideration. It just receives a single word and
  returns words that are the closest in terms of Levenshtein distance.
  It also breaks ties in the edit distance by the number of appearances 
  of each candidate in the original corpora.
  
  It calculates the Levenshtein distance by a Minimum edit distance 
  algorithm implementation (see below).

- #### Approach #2: Enhanced autocorrect
  This version is slightly enhanced comparing to the Naive approach. 
  When creating our vocabulary, we keep pairs of words that appear together
  along with the frequency that they are encountered (together) in the corpus.
  Then we provide pairs of words, where the first word is spelled correctly
  and the second word is misspelled ask the script to give us up to 3 most 
  probable suggestions (if they exist), in case of a misspelling.

  What makes this implementation "Enhanced" is that it takes the contex of the
  word into consideration, by looking at the previous (correct) word. 

  It also calculates the Levenshtein distance by a Minimum edit distance 
  algorithm implementation (see below).


### Minimum edit distance

The model uses the minimum edit distance algorithm to come up with
suggestions. This is a Dynamic Programming algorithm that solves the 
problem quite efficiently. 

We will assume that the only valid edits in a word are:

- Adding a character, with cost 1
- Deleting a character, with cost 1
- Replacing a character, with cost 2 
  (equivalent to an addition followed by a deletion)

More about edit distance can be found 
[here](https://en.wikipedia.org/wiki/Edit_distance).

### Instructions

To avoid copyright issues, I didn't upload any corpora on GitHub. 
To run the script, you just have to create a directory named "books", 
in the same directory  where the script is located and add multiple 
corpora inside as txt files.
