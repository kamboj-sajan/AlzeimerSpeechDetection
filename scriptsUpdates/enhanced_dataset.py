# Enhanced dataset.py with librosa and spaCy feature extraction

from pathlib import Path

import torch
import pandas as pd
import numpy as np
import librosa
import spacy
import re
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, BertTokenizer
import enhanced_config as config

# Load spaCy model for linguistic analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

class EnhancedDementiaDataset(Dataset):
    def __init__(self, metadata_file):
        self.metadata = pd.read_csv(metadata_file)
        if 'transcription' in self.metadata.columns:
            self.metadata['transcription'] = self.metadata['transcription'].fillna("")
        if 'label' in self.metadata.columns:
            self.metadata['label'] = self.metadata['label'].astype(int)
        self.labels = self.metadata['label'].to_numpy()
        # Pitt files are named <participant>-<visit>.wav, so a speaker's recordings share a group
        self.groups = self.metadata['file_path'].map(lambda p: Path(p).stem.split('-')[0]).to_numpy()
        self.prosodic, self.paralinguistic = self._load_or_compute_features()

    def _load_or_compute_features(self):
        """Extracting hand-crafted features is slow, so compute them once and cache them on disk."""
        cache_path = Path(config.FEATURE_CACHE_PATH)
        file_paths = np.array(self.metadata['file_path'].tolist(), dtype=str)
        if cache_path.exists():
            cache = np.load(cache_path)
            if np.array_equal(cache['file_paths'], file_paths):
                return cache['prosodic'], cache['paralinguistic']
            print(f"Feature cache {cache_path} does not match the metadata; recomputing.")

        prosodic, paralinguistic = [], []
        for audio_path, transcription in tqdm(zip(file_paths, self.metadata['transcription']),
                                              total=len(file_paths), desc="Extracting features"):
            prosodic.append(self.extract_prosodic_features(audio_path))
            paralinguistic.append(self.extract_paralinguistic_features(transcription))
        prosodic = np.nan_to_num(np.stack(prosodic)).astype(np.float32)
        paralinguistic = np.nan_to_num(np.stack(paralinguistic)).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, file_paths=file_paths, prosodic=prosodic, paralinguistic=paralinguistic)
        return prosodic, paralinguistic

    def __len__(self):
        return len(self.metadata)
    
    def extract_prosodic_features(self, audio_path, sample_rate=16000):
        """Extract prosodic features using librosa"""
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=sample_rate)
            
            # 1. Pitch (F0) features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                pitch_values.append(pitch)
            
            pitch_values = np.array(pitch_values)
            pitch_values = pitch_values[pitch_values > 0]  # Remove zero values
            
            if len(pitch_values) > 0:
                mean_pitch = np.mean(pitch_values)
                std_pitch = np.std(pitch_values)
                # Jitter (pitch variability)
                jitter = std_pitch / mean_pitch if mean_pitch > 0 else 0
            else:
                mean_pitch = std_pitch = jitter = 0
                
            # 2. Energy features
            rms_energy = librosa.feature.rms(y=y)[0]
            mean_energy = np.mean(rms_energy)
            std_energy = np.std(rms_energy)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_spectral_centroid = np.mean(spectral_centroids)
            
            prosodic_features = np.array([
                mean_pitch,
                std_pitch, 
                jitter,
                mean_energy,
                std_energy,
                mean_spectral_centroid
            ])
            
            return prosodic_features
            
        except Exception as e:
            print(f"Error extracting prosodic features from {audio_path}: {e}")
            # Return zeros for 6 prosodic features
            return np.zeros(6)
    
    def extract_paralinguistic_features(self, transcription):
        """Extract paralinguistic features using spaCy"""
        if not nlp or not transcription:
            return np.zeros(4)  # Return zeros for 4 paralinguistic features
            
        try:
            doc = nlp(transcription)
            
            # 1. Speech rate (words per minute - normalized by assuming 1 minute)
            words = [token for token in doc if not token.is_punct and not token.is_space]
            word_count = len(words)
            speech_rate = word_count  # Normalized speech rate
            
            # 2. Type-Token Ratio (TTR) - lexical diversity
            if word_count > 0:
                unique_words = set([token.lemma_.lower() for token in words if token.is_alpha])
                ttr = len(unique_words) / word_count
            else:
                ttr = 0
                
            # 3. Pause indicators (count of punctuation marks)
            punct_count = len([token for token in doc if token.is_punct])
            pause_rate = punct_count / max(word_count, 1)
            
            # 4. Average sentence length
            sentences = list(doc.sents)
            if sentences:
                avg_sentence_length = sum(len([t for t in sent if not t.is_punct and not t.is_space]) 
                                        for sent in sentences) / len(sentences)
            else:
                avg_sentence_length = 0
                
            paralinguistic_features = np.array([
                speech_rate,
                ttr,
                pause_rate,
                avg_sentence_length
            ])
            
            return paralinguistic_features
            
        except Exception as e:
            print(f"Error extracting paralinguistic features: {e}")
            return np.zeros(4)
    
    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['file_path']
        transcription = self.metadata.iloc[idx]['transcription']
        label = self.metadata.iloc[idx]['label']
        
        try:
            # Load audio for Wav2Vec2: resampled, mono, and capped at MAX_AUDIO_SECONDS
            final_audio, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True,
                                          duration=config.MAX_AUDIO_SECONDS)

            return {
                "raw_audio": final_audio.astype(np.float32),
                "transcription": str(transcription),
                "prosodic_features": self.prosodic[idx],
                "paralinguistic_features": self.paralinguistic[idx],
                "label": label
            }
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}. Returning dummy data.")
            dummy_audio = np.zeros(config.SAMPLE_RATE * 1, dtype=np.float32)
            dummy_prosodic = np.zeros(6, dtype=np.float32)
            dummy_paralinguistic = np.zeros(4, dtype=np.float32)
            
            return {
                "raw_audio": dummy_audio,
                "transcription": "",
                "prosodic_features": dummy_prosodic,
                "paralinguistic_features": dummy_paralinguistic,
                "label": label
            }

def get_processors():
    """Initializes and returns the acoustic processor and linguistic tokenizer."""
    print("Loading Wav2Vec2 processor and BERT tokenizer...")
    acoustic_processor = Wav2Vec2Processor.from_pretrained(config.ACOUSTIC_MODEL_NAME)
    linguistic_tokenizer = BertTokenizer.from_pretrained(config.LINGUISTIC_MODEL_NAME)
    print("Processors loaded.")
    return acoustic_processor, linguistic_tokenizer