def repetition_penalty(generated, last_char, window=40):
    recent = generated[-window:]
    return 2.0 if last_char in recent else 1.0
