from src.utils.syllables import count_syllables

def line_syllables(line):
    return sum(count_syllables(w) for w in line.split())

def cadence_penalty(line, target):
    return abs(line_syllables(line) - target)
