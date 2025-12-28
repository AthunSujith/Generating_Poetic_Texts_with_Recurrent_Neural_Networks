from pathlib import Path

ROOT = Path(".")

STRUCTURE = [
    # Data
    "data/raw",
    "data/cleaned",
    "data/splits",

    # Models
    "models/checkpoints",
    "models/final",

    # Source
    "src/data",
    "src/training",
    "src/inference",
    "src/utils",

    # Experiment management
    "experiments",

    # Logs
    "logs"
]

def main():
    print("Initializing poetry_rnn project structure...\n")

    for folder in STRUCTURE:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")

    # Boilerplate files
    (ROOT / "README.md").touch(exist_ok=True)
    (ROOT / "requirements.txt").touch(exist_ok=True)

    print("\nProject structure successfully created.")

if __name__ == "__main__":
    main()
