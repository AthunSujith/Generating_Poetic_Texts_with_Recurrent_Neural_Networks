# VerseRNN: Poetic Text Generation with Recurrent Neural Networks

VerseRNN is a deep learning project focused on generating poetic texts using character-level Recurrent Neural Networks (LSTMs). It incorporates specialized techniques like **Locked Dropout** (AWD-LSTM inspired) and a unique **Poetic Loss** function that incentivizes rhyming and rhythmic consistency.

## 🌟 Features

-   **Character-level LSTM Architecture**: High-capacity 3-layer LSTM with Layer Normalization and Residual connections.
-   **AWD-LSTM Inspired Regularization**: Implements Locked Dropout (consistent masks across time steps) for superior sequence learning.
-   **Poetic Loss Function**: A customized Cross-Entropy loss that adds a "rhyme bonus" based on phonetic similarity of ending words.
-   **Advanced Decoding**: Supports both standard temperature-based sampling and **Beam Search** for more coherent poetic structures.
-   **Structured Pipeline**: Dedicated scripts for cleaning, splitting, and training on raw text corpora.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/VerseRNN.git
    cd VerseRNN
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Data Preparation
Place your raw poetry text file in `data/raw/poems_v1.txt`. Then run:
```bash
python src/data/clean.py
python src/data/make_splits.py
```

### 2. Training
To train the model from scratch:
```bash
python src/training/train.py
```
Checkpoints will be saved in `models/checkpoints/`.

### 3. Generation
To generate poetry using a trained model:
```bash
# Standard sampling
python src/inference/generate_torch.py

# Beam search sampling (more coherent)
python src/inference/beam_poet.py
```

## 📂 Project Structure

-   `data/`: Raw, cleaned, and split datasets.
-   `models/`: Saved model checkpoints and final weights.
-   `src/data/`: Scripts for data normalization and splitting.
-   `src/training/`: Core model definition, loss functions, and training loop.
-   `src/inference/`: Poetic text generation utilities (Beam search & Sampling).
-   `src/utils/`: Phonic matching, rhyme detection, and syllable counting.

## 🧪 Experiments
Experimental configurations and logs are stored in the `experiments/` and `logs/` directories. These are used to tracking hyperparameter tuning and loss curves.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
