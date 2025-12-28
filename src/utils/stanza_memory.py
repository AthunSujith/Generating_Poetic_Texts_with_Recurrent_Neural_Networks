class StanzaMemory:
    def __init__(self):
        self.lines = []

    def add_line(self, line):
        self.lines.append(line.strip())

    def last_rhyme(self):
        if not self.lines:
            return None
        return self.lines[-1].split()[-1]
