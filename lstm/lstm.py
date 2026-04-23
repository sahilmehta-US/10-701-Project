import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, hidden_size, attention_hidden_size = None, dropout = 0.0):
        super().__init__()
        attention_hidden_size = attention_hidden_size or hidden_size
        self.key = nn.Linear(hidden_size, attention_hidden_size)
        self.query = nn.Linear(hidden_size, attention_hidden_size)
        self.gate = nn.Linear(attention_hidden_size, attention_hidden_size)
        self.score = nn.Linear(attention_hidden_size, 1)
        self.output = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, seq_states, q_state):
        keys = self.key(seq_states)
        query = self.query(q_state).unsqueeze(1)
        alignment = torch.tanh(keys + query)
        gate = torch.sigmoid(self.gate(alignment))
        gated_alignment = self.dropout(alignment * gate)
        scores = self.score(gated_alignment).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * seq_states, dim=1)
        fused = torch.cat([context, q_state], dim=-1)
        fused = self.norm(self.output(fused) + q_state)
        return fused, weights

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.0, use_attn_gate = True, attn_hidden_size = None, attn_dropout = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        if use_attn_gate:
            self.attn_gate = AttentionGate(hidden_size, attn_hidden_size, attn_dropout)
        else:
            self.attn_gate = None
        self.last_attn_weights = None

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
        last_hidden = out[:, -1, :]
        if self.attn_gate is not None:
            pooled, attn_weights = self.attn_gate(out, last_hidden)
            self.last_attn_weights = attn_weights.detach()
            last_hidden = pooled
        # fc output:     (batch, 1)             e.g. (64, 1)
        # after squeeze: (batch,)               e.g. (64,)
        return self.fc(last_hidden).squeeze(-1)
