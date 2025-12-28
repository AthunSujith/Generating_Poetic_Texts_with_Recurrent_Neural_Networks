import numpy as np
from src.utils.rhyme import rhyme_part


def rhyme_bias(probs, chars, rhyme_word, strength=1.5):
    if not rhyme_word:
        return probs
    tail = rhyme_part(rhyme_word)
    for i,c in enumerate(chars):
        if rhyme_part(c) == tail:
            probs[i] *= strength
    return probs / probs.sum()
