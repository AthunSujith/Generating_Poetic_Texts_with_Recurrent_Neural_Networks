import re
from pathlib import Path

RAW_PATH = Path("data/raw/poems_v1.txt")
OUT_PATH = Path("data/cleaned/poems_clean.txt")

def normalize_text(text: str) -> str:
    text = text.lower()

    # Normalize quotes and dashes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")

    # Remove weird symbols (keep letters, punctuation, newline)
    text = re.sub(r"[^a-z0-9\s\n.,;:!?'\-]", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"

def main():
    print("Cleaning poetry corpus...")
    raw = RAW_PATH.read_text(encoding="utf8")
    clean = normalize_text(raw)
    OUT_PATH.write_text(clean, encoding="utf8")
    print("Saved cleaned dataset to:", OUT_PATH)

if __name__ == "__main__":
    main()
