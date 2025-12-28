from pathlib import Path

CLEAN_PATH = Path("data/cleaned/poems_clean.txt")

TRAIN_PATH = Path("data/splits/train.txt")
VALID_PATH = Path("data/splits/valid.txt")

def main():
    text = CLEAN_PATH.read_text(encoding="utf8")
    cut = int(len(text) * 0.9)

    TRAIN_PATH.write_text(text[:cut], encoding="utf8")
    VALID_PATH.write_text(text[cut:], encoding="utf8")

    print("Train/Valid split complete.")

if __name__ == "__main__":
    main()
