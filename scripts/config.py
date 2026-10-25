import torch

DATA_DIR = "../data/pitt" 
OUTPUT_CSV_PATH = "../data/metadata.csv"
MODEL_SAVE_PATH = "../models/best_dementia_model.pth"

# -- Audio & Text Processing --
SAMPLE_RATE = 16000
ACOUSTIC_MODEL_NAME = "facebook/wav2vec2-base-960h"
LINGUISTIC_MODEL_NAME = "bert-base-uncased"

# -- Model Hyperparameters --
# Dimensions from Wav2Vec2-base and BERT-base
ACOUSTIC_DIM = 768
LINGUISTIC_DIM = 768
# Number of co-attention layers (Nx in the paper)
NUM_ATTENTION_LAYERS = 3
# Dropout rate
DROPOUT = 0.1

# -- Training Hyperparameters --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Reduce if you run out of GPU memory
EPOCHS = 30
LEARNING_RATE = 2e-5
# Percentage of data for validation
VALIDATION_SPLIT = 0.15