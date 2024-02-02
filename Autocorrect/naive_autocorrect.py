import os
from collections import Counter

DIRECTORY_PATH = "books"
MAX_DISTANCE_ALLOWED = 3
WORDS = [
    'take', 'tadr', 'mass', 'masr', 'byti', 'dfews', 'enormoug', 'indicage', 'colaborate',
    'enburanse', 'ambulanf', 'desk', 'desvio', 'negotaiite', 'allcoated', 'airplame'
]


def create_vocabulary(directory_path):

    vocabulary = Counter()
    corpus_size = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                corpus = file.read().lower()
                corpus_cleaned = ''.join(char if char.islower() else ' ' for char in corpus)
                words = corpus_cleaned.split()

                vocabulary.update(words)
                corpus_size += len(words)

    # Calculate Probabilities
    for word in vocabulary:
        vocabulary[word] /= corpus_size

    return vocabulary


def minimum_edit_distance(source, vocabulary, maximum_distance_allowed):
    candidates = {}

    len_source = len(source)

    for target in vocabulary:
        len_target = len(target)

        # Reject too long or too short words
        if abs(len_source - len_target) > maximum_distance_allowed:
            continue

        # Initialize dp array for minimum edit distance
        med_list = [[i + j if i == 0 or j == 0 else 0 for j in range(len_target + 1)] for i in range(len_source + 1)]

        # Fill dp array
        for i in range(1, len_source + 1):
            for j in range(1, len_target + 1):
                distance_if_deleting_letter = med_list[i][j-1] + 1
                distance_if_adding_letter = med_list[i-1][j] + 1
                distance_if_replacing_letter = med_list[i-1][j-1] if source[i-1] == target[j-1] else med_list[i-1][j-1] + 2

                med_list[i][j] = min(distance_if_replacing_letter, min(distance_if_adding_letter, distance_if_deleting_letter))

        # Check if the minimum edit distance is within the allowed maximum
        if med_list[-1][-1] <= maximum_distance_allowed:
            candidates[target] = (med_list[-1][-1], vocabulary[target])

    # Sort candidates by probability of appearance in the corpora
    sorted_candidates = sorted(candidates.items(), key=lambda item: (item[1][0], -1 * item[1][1]))

    return sorted_candidates


def autocorrect(word, vocabulary, maximum_distance_allowed):
    if word not in vocabulary:
        candidates = minimum_edit_distance(word, vocabulary, maximum_distance_allowed)

        print(f"Candidates for word '{word}':")

        if len(candidates):
            for _, (candidate, (edit_distance, probability)) in enumerate(candidates[:3]):
                print(f"\t-'{candidate}' with Edit Distance {edit_distance} and Vocabulary Probability {probability}")
        else:
            print("\t-No candidates found")
    else:
        print(f"Word '{word}' is spelled correctly.")


if __name__ == "__main__":
    vocabulary = create_vocabulary(DIRECTORY_PATH)
    for word in WORDS:
        autocorrect(word, vocabulary, MAX_DISTANCE_ALLOWED)