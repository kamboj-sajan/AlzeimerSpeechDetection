# scripts/dataset.py

import torch
import torchaudio
import pandas as pd
import numpy as np  # Import numpy
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, BertTokenizer
import config

class DementiaDataset(Dataset):
    def __init__(self, metadata_file):
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['file_path']
        transcription = self.metadata.iloc[idx]['transcription']
        label = self.metadata.iloc[idx]['label']

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 1. Resample if necessary
            if sample_rate != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 2. Ensure waveform is mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Squeeze to 1D, convert to NumPy array, and return
            # This is the key change to fix the ValueError
            final_audio = waveform.squeeze(0).numpy()

            return {
                "raw_audio": final_audio,
                "transcription": str(transcription),
                "label": label
            }

        except Exception as e:
            # If anything goes wrong, return a complete set of dummy data
            print(f"Error loading or processing audio file {audio_path}: {e}. Returning dummy data.")
            dummy_audio = np.zeros(config.SAMPLE_RATE * 1, dtype=np.float32) # Dummy 1-second silent audio
            return {
                "raw_audio": dummy_audio,
                "transcription": "", 
                "label": 0 
            }

def get_processors():
    """Initializes and returns the acoustic processor and linguistic tokenizer."""
    print("Loading Wav2Vec2 processor and BERT tokenizer...")
    acoustic_processor = Wav2Vec2Processor.from_pretrained(config.ACOUSTIC_MODEL_NAME)
    linguistic_tokenizer = BertTokenizer.from_pretrained(config.LINGUISTIC_MODEL_NAME)
    print("Processors loaded.")
    return acoustic_processor, linguistic_tokenizer