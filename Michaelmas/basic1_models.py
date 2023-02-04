import torch
import torch.nn as nn

class linear_net(nn.Module):
    def __init__(self, num_features, dropout=0):
        """"Parameter num_features matches input dimension to data."""
        super(linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(100, 200)
        self.linear_3 = nn.Linear(200, 100)
        self.linear_4 = nn.Linear(100, 1)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        scores = self.relu(scores)
        scores = self.dropout(scores)
        scores = self.linear_3(scores)
        scores = self.relu(scores)
        scores = self.linear_4(scores)
        return scores

class small_linear_net(nn.Module):
    def __init__(self, num_features):
        super(small_linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 10)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(10, 1)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        return scores

class medium_linear_net(nn.Module):
    def __init__(self, num_features):
        super(medium_linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 50)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        return scores