import os
from collections import Counter

DIRECTORY_PATH = "books"
MAX_DISTANCE_ALLOWED = 3
WORD_PAIRS = [
    'i take', 'they tadr', 'the mass', 'the masr', 'them byti', 'a dfews', 'was enormoug', 'to indicage',
    'to colaborate', 'the enburanse', 'an ambulanf', 'a desk', 'the desvio', 'to negotaiite', 'then allcoated',
    'an airplame',
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
                words = ['<S>'] + corpus_cleaned.split()

                # Iterate over pairs of consecutive words
                for prev_word, current_word in zip(words, words[1:]):
                    if current_word not in vocabulary:
                        vocabulary[current_word] = Counter({prev_word: 1})
                    else:
                        vocabulary[current_word][prev_word] += 1

                corpus_size += len(words)

    vocabulary = {word: {prev_word: count / corpus_size for prev_word, count in prev_word_counts.items()} for
                  word, prev_word_counts in vocabulary.items()}

    return vocabulary


def minimum_edit_distance(source, previous_word, vocabulary, maximum_distance_allowed):
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

    def sorting_key(candidate):
        word, (edit_distance, inner_counter) = candidate
        probability = inner_counter[previous_word] if previous_word in inner_counter else 0
        total_count = sum(inner_counter.values())

        return (probability, -total_count)

    # Sort candidates by probability of appearance in the corpora
    sorted_candidates = sorted(candidates.items(), key=sorting_key, reverse=True)

    return sorted_candidates


def autocorrect(word, previous_word, vocabulary, maximum_distance_allowed):
    if word not in vocabulary:
        candidates = minimum_edit_distance(word, previous_word, vocabulary, maximum_distance_allowed)

        print(f"Candidates for word '{word}' with previous word '{previous_word}':")

        if len(candidates):
            for _, (candidate, (edit_distance, previous_word_probability)) in enumerate(candidates[:3]):
                if previous_word in previous_word_probability:
                    print(f"\t-'{candidate}' with Edit Distance {edit_distance} and "
                          f"Probability {previous_word_probability[previous_word]} for previous word '{previous_word}'")
                else:
                    print(f"\t-'{candidate}' with Edit Distance {edit_distance} and "
                          f"Total Probability {sum(previous_word_probability.values())} for any previous words")
        else:
            print("\t-No candidates found")
    else:
        print(f"Word '{word}' is spelled correctly.")


if __name__ == "__main__":
    vocabulary = create_vocabulary(DIRECTORY_PATH)
    for word_pair in WORD_PAIRS:
        first_word, second_word = word_pair.split()
        autocorrect(second_word, first_word, vocabulary, MAX_DISTANCE_ALLOWED)
