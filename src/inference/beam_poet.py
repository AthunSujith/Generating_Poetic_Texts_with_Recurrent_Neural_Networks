import torch
import numpy as np
from src.training.model import PoetryRNN
from src.training.dataset import CHARS, CHAR_TO_IX, IX_TO_CHAR, VOCAB_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def beam_generate(seed, beam=5, length=400, temp=1.0):
    model = PoetryRNN(VOCAB_SIZE).to(DEVICE)
    checkpoint_path = "models/final/poetry_rnn_final.pt"
    
    try:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model.")
        
    model.eval()

    beams = [(seed.lower(), None, 0.0)]
    seq_len = 80

    for _ in range(length):
        new_beams = []
        for text, h, score in beams:
            x = torch.zeros(1, seq_len, VOCAB_SIZE).to(DEVICE)
            curr_seq = text[-seq_len:]
            for t, ch in enumerate(curr_seq):
                if ch in CHAR_TO_IX:
                    x[0, t + (seq_len - len(curr_seq)), CHAR_TO_IX[ch]] = 1

            out, h_new = model(x, h)
            logits = out.cpu().numpy() / temp
            
            # Subtract max for numerical stability in softmax
            logits = logits - np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))

            # Select top-k (beam size) candidates
            top_indices = np.argsort(probs)[-beam:]
            for i in top_indices:
                new_text = text + IX_TO_CHAR[i]
                new_score = score + np.log(probs[i] + 1e-10)
                new_beams.append((new_text, h_new, new_score))

        # Keep top beam results
        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam]

    return beams[0][0]

if __name__ == "__main__":
    print(beam_generate("the moon", beam=3, length=100))
