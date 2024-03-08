import os
import re
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Helpers.CBoW import get_embeddings, visualize_embeddings
from cbow_sentiment_analysis_nn import MovieReviewsClassifierNN, MovieReviewsDataset


TRAIN_DIRECTORY_PATH = "aclImdb/train/"
TEST_DIRECTORY_PATH = "aclImdb/test/"
NUM_EPOCHS = 50
LEARNING_RATE = 0.1
LAMBDA = 1
BATCH_SIZE = 256


def read_file_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    data = re.sub(r'[,!?;-]', '.', data)
    data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data if ch.isalpha() or ch == '.']
    return data


def preprocess_data_for_cbow():
    tokenized_data = []
    vocabulary = set()

    # Tokenize data, TRY CLEANING, STEMMING
    for i, subdirectory in enumerate(['pos', 'neg']):
        for filename in os.listdir(os.path.join(TRAIN_DIRECTORY_PATH, subdirectory)):
            if filename.endswith(".txt"):
                file_path = os.path.join(TRAIN_DIRECTORY_PATH, subdirectory, filename)
                tokenized_sentence = read_file_and_tokenize(file_path)
                tokenized_data.append(tokenized_sentence)

                for tk in tokenized_sentence:
                    vocabulary.add(tk)

    int_to_word = sorted(list(vocabulary))
    word_to_int = {word: i for i, word in enumerate(int_to_word)}

    return tokenized_data, int_to_word, word_to_int


def preprocess_data_for_classification(embeddings, word_to_int, directory):
    x = []
    y = []

    # Tokenize data
    for i, subdirectory in enumerate(['pos', 'neg']):
        for filename in os.listdir(os.path.join(directory, subdirectory)):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, subdirectory, filename)
                tokenized_sentence = read_file_and_tokenize(file_path)

                sentence_embedding = np.zeros(embeddings.shape[1])
                word_count = 0

                for word in tokenized_sentence:
                    if word in word_to_int:
                        word_embedding = embeddings[word_to_int[word]]
                        sentence_embedding += word_embedding
                        word_count += 1

                if word_count > 0:
                    sentence_embedding /= word_count

                x.append(sentence_embedding)
                y.append(int(subdirectory == 'pos'))

    x = np.array(x)
    y = np.array(y)

    return x, y


def train_classifier(model, criterion, optimizer, train_loader):
    for epoch in range(NUM_EPOCHS):
        for i, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate_classifier(model, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device).long()

            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        accuracy = 100.0 * n_correct / n_samples

        return accuracy


if __name__ == "__main__":
    tokenized_data, int_to_word, word_to_int = preprocess_data_for_cbow()

    embedding_weights_np = get_embeddings(tokenized_data, word_to_int)
    visualize_embeddings(embedding_weights_np)

    model = MovieReviewsClassifierNN(embedding_weights_np.shape[1], 128, 32, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)  # L2 Regularization

    x_train, y_train = preprocess_data_for_classification(embedding_weights_np, word_to_int, TRAIN_DIRECTORY_PATH)
    train_dataset = MovieReviewsDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    train_classifier(model, criterion, optimizer, train_loader)

    x_test, y_test = preprocess_data_for_classification(embedding_weights_np, word_to_int, TEST_DIRECTORY_PATH)
    test_dataset = MovieReviewsDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    accuracy = evaluate_classifier(model, test_loader)
    print(f"Accuracy of the model for movie reviews classification: {accuracy}%")



