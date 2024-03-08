import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_SIZE = 7
NUM_EPOCHS = 50
LEARNING_RATE = 0.1
LAMBDA = 1
EMBEDDINGS_SIZE = 10
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CBoWCustomNN(nn.Module):

    def __init__(self, vocabulary_size, embeddings_size):
        super(CBoWCustomNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embeddings_size)
        self.output_layer = nn.Linear(embeddings_size, vocabulary_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.long()
        out1 = self.embedding(x)
        embedded_sum = out1.sum(dim=1)
        out1 = self.relu(embedded_sum)

        out2 = self.output_layer(out1)

        return out2


class CBoWDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def preprocess_data(tokenized_data, word_to_int, vocabulary_size):
    x = []
    y = []


    for tokenized_sentence in tokenized_data:
        if len(tokenized_sentence) >= WINDOW_SIZE:
            for i in range(WINDOW_SIZE // 2, len(tokenized_sentence) - WINDOW_SIZE // 2):
                y.append(word_to_int[tokenized_sentence[i]])
                context_words = (tokenized_sentence[i - WINDOW_SIZE // 2:i] + tokenized_sentence[
                                                                              i + 1:i + WINDOW_SIZE // 2 + 1])

                context_words_one_hot_encoding = np.zeros(vocabulary_size)
                for word in context_words:
                    context_words_one_hot_encoding[word_to_int[word]] += 1
                x.append(context_words_one_hot_encoding / (WINDOW_SIZE - 1))

    x = np.array(x)
    y = np.array(y)

    return x, y


def get_embeddings(tokenized_data, word_to_int):
    vocabulary_size = len(word_to_int)
    x_train, y_train = preprocess_data(tokenized_data, word_to_int, vocabulary_size)

    train_dataset = CBoWDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = CBoWCustomNN(vocabulary_size, EMBEDDINGS_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)  # L2 Regularization

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

    embedding_weights = model.embedding.weight.data  # Accessing the weights of the embedding layer
    embedding_weights_np = embedding_weights.cpu().numpy()

    return embedding_weights_np


def visualize_embeddings(embedding_weights_np):
    first_20_embeddings = embedding_weights_np[:20]

    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(first_20_embeddings)

    # Plot the embeddings in a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for i, word in enumerate(int_to_word[:20]):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Word Embeddings Visualization using PCA')
    plt.grid(True)
    plt.show()
