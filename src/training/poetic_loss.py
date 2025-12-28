import torch
import torch.nn.functional as F

from src.utils.rhyme import rhyme_part

def poetic_loss(logits, targets, prev_lines):
    ce = F.cross_entropy(logits, targets)

    # Rhyme bonus for last character of a line
    rhyme_bonus = 0.0
    for i, tgt in enumerate(targets):
        if prev_lines[i]:
            last = prev_lines[i].split()[-1]
            cur  = prev_lines[i].split()[-1]
            if rhyme_part(last) == rhyme_part(cur):
                rhyme_bonus += 0.1

    return ce - rhyme_bonus
