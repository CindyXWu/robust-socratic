import torch
import torch.nn as nn

class linear_net(nn.Module):
    def __init__(self, num_features, dropout=0):
        """"Parameter num_features matches input dimension to data."""
        super(linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 1000)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(1000, 1500)
        self.linear_3 = nn.Linear(1500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        scores = self.relu(scores)
        #scores = self.dropout(scores)
        scores = self.sigmoid(self.linear_3(scores))
        return scores

class small_linear_net(nn.Module):
    def __init__(self, num_features):
        super(small_linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 50)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.sigmoid(self.linear_2(scores))
        return scores