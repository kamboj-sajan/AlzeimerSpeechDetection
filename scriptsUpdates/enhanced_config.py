import torch

# -- Project Paths --
DATA_DIR = "../data/pitt"
OUTPUT_CSV_PATH = "../data/metadata.csv"
MODEL_SAVE_PATH = "../models/best_dementia_model.pth"
# Cached prosodic/paralinguistic features; delete it if you change preprocessing
FEATURE_CACHE_PATH = "../data/handcrafted_features.npz"

# -- Audio & Text Processing --
SAMPLE_RATE = 16000
ACOUSTIC_MODEL_NAME = "facebook/wav2vec2-base-960h"
LINGUISTIC_MODEL_NAME = "bert-base-uncased"
MAX_AUDIO_SECONDS = 30  # Wav2Vec2 memory grows quadratically with length; lower if you hit CUDA OOM
TEXT_MAX_LENGTH = 256   # BERT tokens kept per transcript (max 512)

# -- Model Hyperparameters --
# Dimensions from Wav2Vec2-base and BERT-base
ACOUSTIC_DIM = 768
LINGUISTIC_DIM = 768

# Number of co-attention layers (Nx in the paper)
NUM_ATTENTION_LAYERS = 3

# Dropout rate
DROPOUT = 0.1

# -- Feature Dimensions --
# Prosodic features: pitch_mean, pitch_std, jitter, energy_mean, energy_std, spectral_centroid
PROSODIC_FEATURE_DIM = 6

# Paralinguistic features: speech_rate, ttr, pause_rate, avg_sentence_length  
PARALINGUISTIC_FEATURE_DIM = 4

# -- Training Hyperparameters --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Increased for better stability with enhanced features
EPOCHS = 30     # Reduced due to early stopping
LEARNING_RATE = 1e-5  # Reduced for better convergence (pretrained Wav2Vec2/BERT layers)
HEAD_LEARNING_RATE = 1e-4  # Randomly initialized fusion and classifier layers
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHTS = True
USE_AMP = True

# -- Scheduler Hyperparameters --
SCHEDULER_NAME = "cosine"
SCHEDULER_WARMUP_RATIO = 0.1  # Fraction of total steps used for warmup

# -- Cross-validation Parameters --
K_FOLDS = 10
RANDOM_SEED = 42

# Validation split within each fold (for 8 train, 1 val, 1 test)
TRAIN_RATIO = 8/9   # 8 out of 9 remaining folds for training
VAL_RATIO = 1/9     # 1 out of 9 remaining folds for validation

# -- Early Stopping --
PATIENCE = 5        # Number of epochs to wait for improvement
MIN_DELTA = 0.001   # Minimum improvement threshold

# -- Backbone Fine-tuning --
UNFREEZE_EPOCH = 3
UNFREEZE_ACOUSTIC_LAYERS = 4
UNFREEZE_LINGUISTIC_LAYERS = 4

# -- Results --
RESULTS_DIR = "../results"
MODELS_DIR = "../models"
HISTORY_DIR = "../results/histories"
PLOTS_DIR = "../results/plots"