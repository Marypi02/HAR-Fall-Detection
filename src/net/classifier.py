import torch.nn as nn

class ClassifierBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_features, dropout=0.3, bidirectional=False):
        super(ClassifierBlock, self).__init__()

        input_dim = input_size * 2 if bidirectional else input_size # se bidirectional è True, abbiamo due LSTM e quindi il doppio dei parametri
        hidden_dim = hidden_size // 2 # size per il secondo hidden layer

        self.net = nn.Sequential(
            # --- PRIMO HIDDEN LAYER ---
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),

            # --- SECONDO HIDDEN LAYER
            nn.Linear(hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),

            # --- USCITA ---
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)