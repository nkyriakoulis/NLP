import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MovieReviewsClassifierNN(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MovieReviewsClassifierNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        self.l3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.l1(x)
        out1 = self.relu(out1)
        out2 = self.l2(out1)
        out2 = self.relu(out2)
        out3 = self.l3(out2)

        return out3


class MovieReviewsDataset(Dataset):
    def __init__(self, X, y):
        # Convert the selected columns DataFrame to a PyTorch tensor
        self.x = torch.tensor(X, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
