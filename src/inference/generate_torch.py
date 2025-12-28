import torch
import numpy as np
from src.training.model import PoetryRNN
from src.training.dataset import CHARS, CHAR_TO_IX, IX_TO_CHAR, VOCAB_SIZE

def sample(logits, temp=0.7):
    logits = logits.cpu().numpy() / temp
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(probs), p=probs)

def generate(seed, length=400, temp=0.7):
    model = PoetryRNN(VOCAB_SIZE)
    checkpoint_path = "models/final/poetry_rnn_final.pt"
    
    try:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model.")

    model.eval()

    result = seed.lower()
    h = None
    # We use a window of up to 80 chars (SEQ_LEN)
    seq_len = 80
    
    for _ in range(length):
        x = torch.zeros(1, seq_len, VOCAB_SIZE)
        curr_seq = result[-seq_len:]
        for t, ch in enumerate(curr_seq):
            if ch in CHAR_TO_IX:
                # Align to the right of the input sequence
                x[0, t + (seq_len - len(curr_seq)), CHAR_TO_IX[ch]] = 1

        out, h = model(x, h)
        idx = sample(out, temp)
        result += IX_TO_CHAR[idx]
    return result

if __name__ == "__main__":
    print(generate("the moon", 500, 0.6))
