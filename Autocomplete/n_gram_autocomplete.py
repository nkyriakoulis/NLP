import random
from sklearn.model_selection import train_test_split
import nltk
from collections import defaultdict, Counter

K = 1.0  # Laplacian Smoothing


def tokenize_sentences(sentences):
    return [nltk.word_tokenize(sentence.lower()) for sentence in sentences]


def create_closed_vocabulary(data, threshold):
    vocabulary = defaultdict(int)

    for sentence in data:
        for word in sentence:
            vocabulary[word] += 1

    closed_vocabulary = {word for word, count in vocabulary.items() if count >= threshold}
    closed_vocabulary |= {'<unk>', '</s>'}

    return closed_vocabulary


def replace_oov_words_by_unk(tokenized_sentences, vocabulary):
    return [[word if word in vocabulary else '<unk>' for word in sentence] for sentence in tokenized_sentences]


def preprocess_data():
    with open("en_US.twitter.txt", "r", encoding="utf-8") as f:
        sentences = f.readlines()

    tokenized_sentences = tokenize_sentences(sentences)
    train_data, test_data = train_test_split(tokenized_sentences, test_size=0.01, random_state=42)

    closed_vocabulary = create_closed_vocabulary(train_data, 2)

    train_data = replace_oov_words_by_unk(train_data, closed_vocabulary)
    test_data = replace_oov_words_by_unk(test_data, closed_vocabulary)

    return train_data, test_data, closed_vocabulary


def count_n_grams(data, n):
    n_grams = Counter()

    for sentence in data:
        padded_sentence = ['<s>'] * n + sentence + ['</s>']

        for i in range(0, len(padded_sentence ) - n + 1):
            n_gram = tuple(padded_sentence [i: i+n])
            n_grams[n_gram] += 1

    return n_grams


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary):
    probabilities = {}

    n = len(vocabulary)
    previous_n_gram_count = n_gram_counts[tuple(previous_n_gram)]

    for word in vocabulary:
        n_plus_1_gram = tuple(previous_n_gram + [word])
        n_plus_1_gram_count = n_plus_1_gram_counts[n_plus_1_gram]

        probability = (n_plus_1_gram_count + K) / (previous_n_gram_count + n * K)
        probabilities[word] = probability

    return probabilities


def suggest_next_word(previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary, starts_with="", top_n=3):

    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary)

    def custom_sort_key(item):
        key, value = item
        if key.startswith(starts_with):
            return (0, -value)  # Prioritize keys starting with "starts_with_" and sort by value in descending order
        else:
            return (1, -value)  # Sort non-prefixed keys by value in descending order

    sorted_probabilities = (sorted(probabilities.items(), key=custom_sort_key))
    return sorted_probabilities[:top_n]


if __name__ == "__main__":
    train_data, test_data, vocabulary = preprocess_data()

    trigram_counts = count_n_grams(train_data, 3)
    four_gram_counts = count_n_grams(train_data, 4)

    for sentence in test_data:
        s = ['<s>'] + sentence

        if len(s) > 3:
            i = random.randint(3, len(s)-1)
            suggestions = suggest_next_word(s[i-3:i], trigram_counts, four_gram_counts, vocabulary,
                                            s[i][:2] if len(s[i]) > 2 else s[i])

            print(f"Top suggestions for completing '{' '.join(s[i-3:i]) + ' ' + s[i][:2] if len(s[i]) > 2 else s[i]}"
                  f"...' sequence are: ")
            for i in range(len(suggestions)):
                print(f"\t-'{suggestions[i][0]}' with probability {100 * suggestions[i][1]:.2f}%")
