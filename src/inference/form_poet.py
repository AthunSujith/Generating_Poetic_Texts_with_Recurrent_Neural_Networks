import torch, numpy as np
from src.training.model import PoetryRNN
from src.training.dataset import VOCAB_SIZE, CHARS, CHAR_TO_IX, IX_TO_CHAR
from src.utils.stanza_memory import StanzaMemory
from src.utils.rhyme_sampler import rhyme_bias
from src.utils.cadence import cadence_penalty
from src.utils.forms import FORMS
from src.utils.emotion_sampler import emotion_bias

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 80

def generate_form(seed, form="haiku", emotion="peace", beam=5, temp=0.7):

    model = PoetryRNN(VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load("models/final/poetry_rnn_final.pt", map_location=DEVICE))
    model.eval()

    targets = FORMS[form]
    memory = StanzaMemory()
    poem = []

    for target in targets:
        beams = [(seed[-SEQ_LEN:], None, 0.0)]

        for _ in range(60):
            new = []
            for text,h,score in beams:

                x = torch.zeros(1, SEQ_LEN, VOCAB_SIZE).to(DEVICE)
                for t,ch in enumerate(text[-SEQ_LEN:]):
                    if ch in CHAR_TO_IX:
                        x[0,t,CHAR_TO_IX[ch]] = 1

                out,h2 = model(x,h)
                probs = torch.softmax(out[0]/temp,0).detach().cpu().numpy()


                probs = emotion_bias(
                            rhyme_bias(probs, CHARS, memory.last_rhyme()),
                            CHARS, emotion
                        )

                for i in np.argsort(probs)[-beam:]:
                    new.append((text + CHARS[i], h2, score + np.log(probs[i] + 1e-8)))

            beams = sorted(new, key=lambda x: x[2], reverse=True)[:beam]

        best = min(beams, key=lambda b: cadence_penalty(b[0], target))[0]
        poem.append(best)
        memory.add_line(best)

    return "\n".join(poem)



print(generate_form("the moon", form="haiku", emotion="sadness"))