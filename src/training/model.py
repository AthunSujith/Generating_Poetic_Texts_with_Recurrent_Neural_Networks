import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        return x * mask


import torch
import torch.nn as nn

class PoetryRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.lockdrop = LockedDropout(0.35)

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=512,
            num_layers=3,
            dropout=0.0,          # turn OFF internal dropout
            batch_first=True
        )

        self.norm = nn.LayerNorm(512)
        self.residual = nn.Linear(vocab_size, 512)

        self.memory_gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )

    def forward(self, x, h=None):

        x = self.lockdrop(x)        # patient dropout on input
        out, h = self.lstm(x, h)
        out = self.lockdrop(out)   # patient dropout on hidden stream

        res = self.residual(x)
        out = self.norm(out + res)

        gate = self.memory_gate(out)
        out = out * gate + 0.05 * torch.randn_like(out)

        out = self.fc(out[:, -1])
        return out, h
