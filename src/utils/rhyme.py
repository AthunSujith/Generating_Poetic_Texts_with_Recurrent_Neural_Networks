import re

VOWELS = "aeiou"

def rhyme_part(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    for i in range(len(word)):
        if word[i] in VOWELS:
            return word[i:]
    return word
