import torch
import torch.nn as nn

class linear_net(nn.Module):
    def __init__(self, num_features, dropout=0.5):
        """"Parameter num_features matches input dimension to data."""
        super(linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_3 = nn.Linear(100, 1)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        scores = self.relu(scores)
        scores = self.dropout(scores)
        scores = nn.Sigmoid(self.linear_3(scores))
        return scores

class small_linear_net(nn.Module):
    def __init__(self, num_features):
        super(small_linear_net, self).__init__()
        self.linear_1 = nn.Linear(num_features, 50)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(50, 1)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = nn.Sigmoid(self.linear_2(scores))
        return scores

# test_model = small_linear_net(2)
# # output model summary statistics
# print(test_model)