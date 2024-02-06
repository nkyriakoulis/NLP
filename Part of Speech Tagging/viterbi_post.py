import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from collections import defaultdict

EPSILON = 0.1  # For Laplacian smoothing
penn_treebank = treebank

tags = set(tag for word, tag in penn_treebank.tagged_words())
tags.add('pi')  # Tag for start ('<s>') token
tags.add('UNK')  # Tag for unknown words

int_to_tag = list(tags)  # Maps integers to tags
tag_to_int = {tag: i for i, tag in enumerate(int_to_tag)}  # Maps tags to integers

vocabulary = set()


class ViterbiSolver:

    def __init__(self, transition_matrix, emission_matrix):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix

        self.sentence = None
        self.probability_matrix = None
        self.backpointer_matrix = None

        self.total_words = 0
        self.correct_tags = 0

    def solve(self, sentence):
        self.sentence = sentence

        self.probability_matrix = [[0 for _ in range(len(self.sentence))] for _ in range(len(tags))]
        self.backpointer_matrix = [[0 for _ in range(len(self.sentence))] for _ in range(len(tags))]

        # Initialize first column of the probability matrix
        for i, tag in enumerate(tags):
            self.probability_matrix[i][0] = self.transition_matrix['pi'][tag]
            if sentence[0][0].lower() in vocabulary:
                self.probability_matrix[i][0] *= self.emission_matrix[tag][self.sentence[0][0].lower()]

        self.forward()
        self.backward()

    def forward(self):
        for j in range(1, len(self.sentence)):
            in_vocabulary = sentence[j][0].lower() in vocabulary

            for current_tag_index, current_tag in enumerate(tags):
                max_probability = 0
                argmax_previous_tag = 0

                for previous_tag_index, previous_tag in enumerate(tags):
                    transition_prob = self.transition_matrix[int_to_tag[previous_tag_index]][
                        int_to_tag[current_tag_index]]

                    if in_vocabulary:
                        emission_prob = self.emission_matrix[int_to_tag[current_tag_index]][sentence[j][0].lower()]
                        prob = self.probability_matrix[previous_tag_index][j - 1] * transition_prob * emission_prob
                    else:
                        prob = self.probability_matrix[previous_tag_index][j - 1] * transition_prob

                    if prob > max_probability:
                        max_probability = prob
                        argmax_previous_tag = previous_tag_index

                self.probability_matrix[current_tag_index][j] = max_probability
                self.backpointer_matrix[current_tag_index][j] = argmax_previous_tag

    def backward(self):
        ans = []
        max_prob = 0
        max_index = 0

        for i in range(len(tags)):
            if self.probability_matrix[i][-1] > max_prob:
                max_prob = self.probability_matrix[i][-1]
                max_index = i

        for i in range(len(self.sentence) - 1, -1, -1):
            ans.append(int_to_tag[max_index])
            max_index = self.backpointer_matrix[max_index][i]

        ans.reverse()

        self.total_words += len(self.sentence)
        for i, (word, tag) in enumerate(self.sentence):
            if ans[i] == tag:
                self.correct_tags += 1
            # Uncomment to do error analysis
            # else:
            #     print(f"Mistakenly tagged '{word}', '{tag}' as '{ans[i]}'")

    def print_results(self):
        print(f"Accuracy: {100 * self.correct_tags / self.total_words :.2f}%")
        print(f"Predicted: {self.correct_tags} correctly out of {self.total_words} words.")


def convert_to_probabilities(dictionary, total):
    # Apply Laplacian Smoothing and convert to probabilities
    for key in dictionary:
        row_sum = sum(dictionary[key].values())
        for val in dictionary[key]:
            dictionary[key][val] = (
                    (dictionary[key][val] + EPSILON) / (row_sum + total * EPSILON))

    return dictionary


def create_transition_matrix(sentences):
    transition_matrix = {}

    for cur_tag in tags:
        transition_matrix[cur_tag] = {}
        for next_tag in tags:
            transition_matrix[cur_tag][next_tag] = 0.0

    for sentence in sentences:
        # Add a start token to count frequencies for the starting PoS tag
        sentence = [('<s>', 'pi')] + sentence

        # Count frequencies
        for i in range(len(sentence) - 1):
            cur_tag = sentence[i][1]
            next_tag = sentence[i + 1][1]
            transition_matrix[cur_tag][next_tag] += 1.0

    return convert_to_probabilities(transition_matrix, len(tags))


def create_emission_matrix(sentences):
    words = set(word.lower() for sentence in sentences for word, tag in sentence)
    words.add('<s>')
    for word in words:
        vocabulary.add(word)

    emission_matrix = {}

    for cur_tag in tags:
        emission_matrix[cur_tag] = {}
        for word in words:
            emission_matrix[cur_tag][word] = 0.0

    for sentence in sentences:
        # Add a start token to count frequencies for the starting PoS tag
        sentence = [('<s>', 'pi')] + sentence

        # Count frequencies
        for word, tag in sentence:
            emission_matrix[tag][word.lower()] += 1.0

    return convert_to_probabilities(emission_matrix, len(words))


if __name__ == "__main__":
    train_data, test_data = train_test_split(penn_treebank.tagged_sents(), test_size=0.2, random_state=42)
    # print(f"Number of training sentences: {len(train_data)}")
    # print(f"Number of testing sentences: {len(test_data)}")

    transition_matrix = create_transition_matrix(train_data)
    emission_matrix = create_emission_matrix(train_data)

    viterbi_solver = ViterbiSolver(transition_matrix, emission_matrix)
    for sentence in test_data:
        viterbi_solver.solve(sentence)

    viterbi_solver.print_results()





