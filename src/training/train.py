import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from src.training.dataset import VOCAB_SIZE, PoetryDataset
from src.training.model import PoetryRNN
from src.training.poetic_loss import poetic_loss
from pathlib import Path

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 70
BATCH = 64
LEARNING_RATE = 3e-4

def train():
    print(f"Using device: {DEVICE}")
    
    # Initialize datasets and loaders
    try:
        train_ds = PoetryDataset("train")
        val_ds   = PoetryDataset("valid")
    except FileNotFoundError:
        print(f"Error: Dataset not found. Please run src/data/clean.py and src/data/make_splits.py first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH)

    # Initialize model, optimizer, and scaler
    model = PoetryRNN(VOCAB_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

    def fake_prev_lines(batch_size):
        # placeholder until we add real stanza memory
        return [""] * batch_size

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()

            prev_lines = fake_prev_lines(len(y))

            with autocast():
                out, _ = model(x)
                loss = poetic_loss(out, y, prev_lines)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Validation
        model.eval()
        vloss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                prev_lines = fake_prev_lines(len(y))
                out, _ = model(x)
                vloss += poetic_loss(out, y, prev_lines).item()

        checkpoint_path = f"models/checkpoints/poetry_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {vloss/len(val_loader):.4f}")

if __name__ == "__main__":
    train()
