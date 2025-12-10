import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    """
    Blocco LSTM modulare. 
    Progettato per prendere in input sequenze e restituire l'output 
    dell'ultimo passo temporale, tipico per la classificazione/regressione.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMBlock, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0.0, # il dropout ha efficacia solo con num_layer > 1 perchè viene eseguito tra layers
            batch_first=True, 
            bidirectional=bidirectional
        )

        self.dropout_final_layer = nn.Dropout(p=dropout) # userà lo 0.1 nel file yaml

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)        
        
        if self.bidirectional:
            
            final_hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            final_hn = hn[-1, :, :]

        final_hn = self.dropout_final_layer(final_hn)
            
        return final_hn