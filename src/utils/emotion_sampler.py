from src.utils.emotions import EMOTION_VECTORS
from src.utils.phonetics import BRIGHT, SOFT, DARK
import numpy as np

def emotion_bias(probs, chars, emotion="peace"):
    weights = EMOTION_VECTORS[emotion]
    for i,c in enumerate(chars):
        if c in BRIGHT:
            probs[i] *= weights["bright"]
        if c in SOFT:
            probs[i] *= weights["soft"]
        if c in DARK:
            probs[i] *= weights["dark"]
    return probs / probs.sum()
