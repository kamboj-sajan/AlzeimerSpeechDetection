import os
import re
import pandas as pd
from tqdm import tqdm
import config

def parse_cha_file(file_path):
    """
    Parses a .cha file to extract and concatenate all lines spoken by the participant (*PAR).
    It cleans the text by removing CHAT annotations, timestamps, and other noise.
    """
    participant_speech = []
    try:
        # Use 'errors=ignore' for robustness against potential encoding issues
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Process only lines from the Participant
                if line.startswith('*PAR:'):
                    # Extract text after the first tab, which separates the speaker tag from the speech
                    parts = line.strip().split('\t', 1)
                    if len(parts) < 2:
                        continue
                    
                    text = parts[1]
                    
                    # --- Text Cleaning using Regular Expressions ---
                    # Remove timestamps like _
                    text = re.sub(r'\s*\d+_\d+\s*', '', text)
                    # Remove CHAT annotations like [+ exc], [//], etc.
                    text = re.sub(r'\[.*?\]', '', text)
                    # Remove sound events like &=laugh
                    text = re.sub(r'&=\S+', '', text)
                    # Remove pauses and other parenthesized annotations
                    text = re.sub(r'\(\..*?\)', '', text)
                    # Remove incomplete utterance markers
                    text = text.replace('+...', '')
                    # Remove word-level annotations like @o
                    text = re.sub(r'@\w', '', text)
                    
                    # Normalize whitespace to a single space
                    clean_text = ' '.join(text.split()).strip()
                    
                    # Add the cleaned text if it's not empty
                    if clean_text and clean_text != ".":
                         participant_speech.append(clean_text)

    except FileNotFoundError:
        print(f"Warning: CHA file not found at {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error reading or parsing CHA file {file_path}: {e}")
        return None
    
    # Join all utterances into a single transcription string
    return " ".join(participant_speech)

def create_metadata_file():
    """
    Walks through the Pitt dataset directory, finds audio/transcription pairs,
    assigns labels, and saves the metadata to a CSV file.
    """
    print("Starting preprocessing for the Pitt dataset...")
    pitt_dir = config.DATA_DIR
    metadata = []
    
    # Check if the base directory exists
    if not os.path.isdir(pitt_dir):
        print(f"\n!!! ERROR: Data directory not found at {os.path.abspath(pitt_dir)}")
        print("Please update the DATA_DIR path in config.py")
        return

    # Process both 'control' and 'dementia' groups
    for group in ['control', 'dementia']:
        print(f"\nProcessing group: {group}")
        
        audio_folder = os.path.join(pitt_dir, group, 'audio')
        transcript_folder = os.path.join(pitt_dir, group, 'transcripts')
        
        if not os.path.isdir(audio_folder):
            print(f"Warning: Audio folder not found for '{group}' at {audio_folder}")
            continue

        # Assign label: 0 for control, 1 for dementia
        label = 0 if group == 'control' else 1
        
        # Iterate over all audio files in the folder
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
        for audio_filename in tqdm(audio_files, desc=f"Finding pairs for {group}"):
            base_name = os.path.splitext(audio_filename)[0]
            audio_path = os.path.join(audio_folder, audio_filename)
            cha_path = os.path.join(transcript_folder, base_name + '.cha')
            
            # Parse the corresponding .cha file to get the transcription
            transcription = parse_cha_file(cha_path)
            
            # Only add to metadata if transcription was successfully extracted
            if transcription:
                metadata.append({
                    'file_path': audio_path,
                    'transcription': transcription,
                    'label': label
                })

    if not metadata:
        print("\n!!! CRITICAL WARNING: No data was processed.")
        print("Please check your directory structure and ensure it matches:")
        print(f"{pitt_dir}/")
        print("├── control/")
        print("│   ├── audio/*.wav")
        print("│   └── transcripts/*.cha")
        print("└── dementia/")
        print("    ├── audio/*.wav")
        print("    └── transcripts/*.cha")
        return
        
    # Create a DataFrame and save it to the specified CSV path
    df = pd.DataFrame(metadata)
    df.to_csv(config.OUTPUT_CSV_PATH, index=False)
    
    print(f"\nPreprocessing complete. Metadata saved to {config.OUTPUT_CSV_PATH}")
    print(f"Total processed files: {len(df)}")

if __name__ == "__main__":
    create_metadata_file()