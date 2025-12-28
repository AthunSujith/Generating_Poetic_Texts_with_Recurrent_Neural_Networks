import torch
from torch.utils.data import Dataset
from pathlib import Path

SEQ_LEN = 80
STEP = 3

# Build GLOBAL vocab from full corpus
FULL_TEXT = Path("data/cleaned/poems_clean.txt").read_text(encoding="utf8")
CHARS = sorted(list(set(FULL_TEXT)))
CHAR_TO_IX = {c:i for i,c in enumerate(CHARS)}
IX_TO_CHAR = {i:c for i,c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

class PoetryDataset(Dataset):
    def __init__(self, split="train"):
        text = Path(f"data/splits/{split}.txt").read_text(encoding="utf8")
        self.data = []

        for i in range(0, len(text) - SEQ_LEN, STEP):
            self.data.append((text[i:i+SEQ_LEN], text[i+SEQ_LEN]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, nxt = self.data[idx]
        x = torch.zeros(SEQ_LEN, VOCAB_SIZE)
        for t,ch in enumerate(seq):
            x[t, CHAR_TO_IX[ch]] = 1
        y = CHAR_TO_IX[nxt]
        return x, y
