import os
from collections import defaultdict
from math import log
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


TRAIN_DIRECTORY_PATH = "aclImdb/train/"
TEST_DIRECTORY_PATH = "aclImdb/test/"
STOP_WORDS = set(stopwords.words('english') + [' '])


def stemming(words):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def read_file_and_return_as_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read().lower()
        corpus_cleaned = ''.join(char if char.islower() else ' ' for char in corpus)
        return corpus_cleaned.split()


class NaiveBayesClassifier:

    def __init__(self, stem=False):
        self.stem = stem
        self.log_likelihood = {}
        self.accuracy_counters = defaultdict(int)

    def train(self, directory_path):
        print("Training...\nThis might take some time...")
        frequencies = {}
        vocabulary_size = 0

        for i, subdirectory in enumerate(['pos', 'neg']):
            for filename in os.listdir(os.path.join(directory_path, subdirectory)):
                if filename.endswith(".txt"):
                    file_path = os.path.join(directory_path, subdirectory, filename)
                    words = read_file_and_return_as_string(file_path)

                    if self.stem:
                        words = stemming(words)

                    for word in words:
                        if word in STOP_WORDS:
                            continue

                        if word not in frequencies:
                            frequencies[word] = [0, 0]
                            vocabulary_size += 1

                        frequencies[word][i] += 1

        # Apply Laplacian Smoothing while calculating probabilities
        for i in range(2):
            class_sum = sum([frequencies[word][i] for word in frequencies])
            for word in frequencies:
                frequencies[word][i] = (frequencies[word][i] + 1) / (class_sum + vocabulary_size)

        # Calculate log likelihood λ for each word
        self.log_likelihood = {word: log(frequencies[word][0] / frequencies[word][1]) for word in frequencies}

        print("Finished Training!\n")

    def predict(self, directory_path):
        print("Predicting...\nThis might take some time...\n")

        for subdirectory in ['pos', 'neg']:
            for filename in os.listdir(os.path.join(directory_path, subdirectory)):
                if filename.endswith(".txt"):
                    file_path = os.path.join(directory_path, subdirectory, filename)
                    words = read_file_and_return_as_string(file_path)

                    if self.stem:
                        words = stemming(words)

                    # Defaults 0 when word not in log_likelihood dictionary
                    review_likelihood = sum(self.log_likelihood.get(word, 0) for word in words if word not in STOP_WORDS)

                    if review_likelihood > 0:
                        self.accuracy_counters['correct_positive'] += (subdirectory == 'pos')
                        self.accuracy_counters['negative_mistaken_as_positive'] += (subdirectory == 'neg')
                    elif review_likelihood < 0:
                        self.accuracy_counters['correct_negative'] += (subdirectory == 'neg')
                        self.accuracy_counters['positive_mistaken_as_negative'] += (subdirectory == 'pos')
                    else:
                        self.accuracy_counters['positive_mistaken_as_neutral'] += (subdirectory == 'pos')
                        self.accuracy_counters['negative_mistaken_as_neutral'] += (subdirectory == 'neg')

                    self.accuracy_counters['total_positive'] += (subdirectory == 'pos')
                    self.accuracy_counters['total_negative'] += (subdirectory == 'neg')

        self.print_results()

    def print_results(self):
        # Print Results
        print(f"Positive Reviews: {self.accuracy_counters['total_positive']}. Classified as:")
        print(
            f"\t-Positive: {self.accuracy_counters['correct_positive']} "
            f"({100 * self.accuracy_counters['correct_positive'] / self.accuracy_counters['total_positive']:.2f}%)"
        )
        print(
            f"\t-Negative: {self.accuracy_counters['positive_mistaken_as_negative']} "
            f"({100 * self.accuracy_counters['positive_mistaken_as_negative'] / self.accuracy_counters['total_positive']:.2f}%)"
        )
        print(
            f"\t-Neutral: {self.accuracy_counters['positive_mistaken_as_neutral']} "
            f"({100 * self.accuracy_counters['positive_mistaken_as_neutral'] / self.accuracy_counters['total_positive']:.2f}%)"
        )

        print(f"Negative Reviews: {self.accuracy_counters['total_negative']}. Classified as:")
        print(
            f"\t-Negative: {self.accuracy_counters['correct_negative']} "
            f"({100 * self.accuracy_counters['correct_negative'] / self.accuracy_counters['total_negative']:.2f}%)"
        )
        print(
            f"\t-Positive: {self.accuracy_counters['negative_mistaken_as_positive']} "
            f"({100 * self.accuracy_counters['negative_mistaken_as_positive'] / self.accuracy_counters['total_negative']:.2f}%)"
        )
        print(
            f"\t-Neutral: {self.accuracy_counters['negative_mistaken_as_neutral']} "
            f"({100 * self.accuracy_counters['negative_mistaken_as_neutral'] / self.accuracy_counters['total_negative']:.2f}%)"
        )

        print()
        print(f"Total Reviews: {self.accuracy_counters['total_positive'] + self.accuracy_counters['total_negative']}.")

        total_correct = self.accuracy_counters['correct_positive'] + self.accuracy_counters['correct_negative']
        total = self.accuracy_counters['total_positive'] + self.accuracy_counters['total_negative']
        print(f"Total Accuracy: {100 * total_correct / total:.2f}%")


if __name__ == "__main__":
    print("Naive Bayes Classifier performing Sentiment Analysis for Movie Reviews")

    for stem in range(2):
        print(f"Performance of the model {'with' if stem else 'without'} stemming")
        nbc = NaiveBayesClassifier(bool(stem))
        nbc.train(TRAIN_DIRECTORY_PATH)
        nbc.predict(TEST_DIRECTORY_PATH)
