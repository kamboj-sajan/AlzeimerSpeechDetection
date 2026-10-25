import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import config
from dataset import DementiaDataset, get_processors
from model import FFGModel

class DataCollator:
    def __init__(self, acoustic_processor, linguistic_tokenizer):
        self.acoustic_processor = acoustic_processor
        self.linguistic_tokenizer = linguistic_tokenizer

    def __call__(self, features):
        raw_audio = [f["raw_audio"] for f in features]
        texts = [f["transcription"] for f in features]
        labels = [f["label"] for f in features]

        acoustic_batch = self.acoustic_processor(
            raw_audio,
            sampling_rate=config.SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        )

        linguistic_batch = self.linguistic_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        labels_batch = torch.tensor(labels, dtype=torch.long)

        return {
            "acoustic_input_values": acoustic_batch.input_values,
            "linguistic_input_ids": linguistic_batch.input_ids,
            "linguistic_attention_mask": linguistic_batch.attention_mask,
            "label": labels_batch
        }

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        acoustic_input = batch['acoustic_input_values'].to(device)
        linguistic_ids = batch['linguistic_input_ids'].to(device)
        linguistic_mask = batch['linguistic_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(acoustic_input, linguistic_ids, linguistic_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            acoustic_input = batch['acoustic_input_values'].to(device)
            linguistic_ids = batch['linguistic_input_ids'].to(device)
            linguistic_mask = batch['linguistic_attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(acoustic_input, linguistic_ids, linguistic_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, accuracy, f1

def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    if not os.path.exists(config.OUTPUT_CSV_PATH):
        print("Metadata file not found. Please run preprocess.py first.")
        return

    acoustic_processor, linguistic_tokenizer = get_processors()
    full_dataset = DementiaDataset(config.OUTPUT_CSV_PATH)

    val_size = int(len(full_dataset) * config.VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    data_collator = DataCollator(acoustic_processor, linguistic_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    model = FFGModel().to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f} | Validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved to {config.MODEL_SAVE_PATH} with F1-score: {best_val_f1:.4f}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
