import torch.nn as nn

class ClassifierBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_features, dropout=0.3, bidirectional=False):
        super(ClassifierBlock, self).__init__()

        input_dim = input_size * 2 if bidirectional else input_size # se bidirectional è True, abbiamo due LSTM e quindi il doppio dei parametri

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x):
        return self.net(x)