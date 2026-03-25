import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x:   (batch, seq_len, input_size)   e.g. (64, 20, 10)
        # h0:  (num_layers, batch, hidden_size) e.g. (2, 64, 64)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # c0:  (num_layers, batch, hidden_size) e.g. (2, 64, 64)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # out: (batch, seq_len, hidden_size)    e.g. (64, 20, 64)  — h at every time step
        # _:   tuple of (h_n, c_n), each (num_layers, batch, hidden_size) — final states, unused
        out, _ = self.lstm(x, (h0, c0))
        # out[:, -1, :]: (batch, hidden_size)   e.g. (64, 64)      — h at last time step only
        # fc output:     (batch, 1)             e.g. (64, 1)
        # after squeeze: (batch,)               e.g. (64,)
        return self.fc(out[:, -1, :]).squeeze(-1)
