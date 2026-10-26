# Enhanced train.py with 10-fold cross-validation

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import get_cosine_schedule_with_warmup

import enhanced_config as config
from enhanced_dataset import EnhancedDementiaDataset, get_processors
from enhanced_model import EnhancedFFGModel

def fit_standardizer(values):
    """Per-feature (mean, std) as tensors; constant features get std 1 so they map to 0."""
    mean = torch.from_numpy(values.mean(axis=0))
    std = torch.from_numpy(values.std(axis=0))
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std

class EnhancedDataCollator:
    def __init__(self, acoustic_processor, linguistic_tokenizer, feature_stats):
        self.acoustic_processor = acoustic_processor
        self.linguistic_tokenizer = linguistic_tokenizer
        # {"prosodic": (mean, std), "paralinguistic": (mean, std)}, fitted on the training split
        self.feature_stats = feature_stats

    def __call__(self, features):
        raw_audio = [f["raw_audio"] for f in features]
        texts = [f["transcription"] for f in features]
        prosodic_features = [f["prosodic_features"] for f in features]
        paralinguistic_features = [f["paralinguistic_features"] for f in features]
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
            max_length=config.TEXT_MAX_LENGTH,
            return_tensors="pt"
        )
        acoustic_values = acoustic_batch.input_values
        prosodic_batch = torch.stack([torch.from_numpy(f) for f in prosodic_features])
        paralinguistic_batch = torch.stack([torch.from_numpy(f) for f in paralinguistic_features])
        # Raw features span ~0.01 (TTR) to ~2000 (spectral centroid); standardize so none dominates
        prosodic_mean, prosodic_std = self.feature_stats["prosodic"]
        paralinguistic_mean, paralinguistic_std = self.feature_stats["paralinguistic"]
        prosodic_batch = (prosodic_batch - prosodic_mean) / prosodic_std
        paralinguistic_batch = (paralinguistic_batch - paralinguistic_mean) / paralinguistic_std
        labels_batch = torch.tensor(labels, dtype=torch.long)
        
        return {
            "acoustic_input_values": acoustic_values,
            "linguistic_input_ids": linguistic_batch.input_ids,
            "linguistic_attention_mask": linguistic_batch.attention_mask,
            "prosodic_features": prosodic_batch,
            "paralinguistic_features": paralinguistic_batch,
            "label": labels_batch
        }

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    use_amp = config.USE_AMP and device.type == "cuda"
    skipped_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        acoustic_input = batch['acoustic_input_values'].to(device)
        linguistic_ids = batch['linguistic_input_ids'].to(device)
        linguistic_mask = batch['linguistic_attention_mask'].to(device)
        prosodic_features = batch['prosodic_features'].to(device)
        paralinguistic_features = batch['paralinguistic_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(acoustic_input, linguistic_ids, linguistic_mask, 
                          prosodic_features, paralinguistic_features)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            # Stepping on a NaN loss would write NaN into every trainable weight
            skipped_batches += 1
            continue
        scaler.scale(loss).backward()
        if config.GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    
    if skipped_batches:
        print(f"Warning: skipped {skipped_batches}/{len(dataloader)} training batches with a non-finite loss")
    if skipped_batches == len(dataloader):
        raise RuntimeError("Every training batch had a non-finite loss; check the input data and model outputs.")
    avg_loss = total_loss / (len(dataloader) - skipped_batches)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return avg_loss, accuracy, f1, precision, recall, auc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    use_amp = config.USE_AMP and device.type == "cuda"
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            acoustic_input = batch['acoustic_input_values'].to(device)
            linguistic_ids = batch['linguistic_input_ids'].to(device)
            linguistic_mask = batch['linguistic_attention_mask'].to(device)
            prosodic_features = batch['prosodic_features'].to(device)
            paralinguistic_features = batch['paralinguistic_features'].to(device)
            labels = batch['label'].to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(acoustic_input, linguistic_ids, linguistic_mask,
                              prosodic_features, paralinguistic_features)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    
    return avg_loss, accuracy, f1, precision, recall, auc, all_preds, all_labels, all_probs

def train_fold(fold, train_indices, val_indices, test_indices, full_dataset, 
               acoustic_processor, linguistic_tokenizer, device):
    """Train a single fold of the 10-fold cross-validation"""
    
    print(f"\n{'='*20} FOLD {fold+1}/10 {'='*20}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")
    
    # Create datasets for this fold
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) 
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create data loaders (feature normalization is fitted on the training split only)
    feature_stats = {
        "prosodic": fit_standardizer(full_dataset.prosodic[train_indices]),
        "paralinguistic": fit_standardizer(full_dataset.paralinguistic[train_indices]),
    }
    data_collator = EnhancedDataCollator(acoustic_processor, linguistic_tokenizer, feature_stats)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, collate_fn=data_collator)
    
    # Initialize model and training components
    model = EnhancedFFGModel().to(device)
    # Pretrained backbones get a small LR; the new fusion/classifier layers need a larger one to learn
    backbone_prefixes = ("acoustic_model.", "linguistic_model.")
    backbone_params = [p for n, p in model.named_parameters() if n.startswith(backbone_prefixes)]
    head_params = [p for n, p in model.named_parameters() if not n.startswith(backbone_prefixes)]
    optimizer = AdamW([
        {"params": backbone_params, "lr": config.LEARNING_RATE},
        {"params": head_params, "lr": config.HEAD_LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        label_counts = pd.Series(full_dataset.labels[train_indices]).value_counts().sort_index()
        total = label_counts.sum()
        weights = total / (len(label_counts) * label_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = config.EPOCHS * steps_per_epoch
    scheduler = None
    if config.SCHEDULER_NAME.lower() == "cosine" and total_steps > 0:
        warmup_steps = int(total_steps * config.SCHEDULER_WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device.type == "cuda")
    
    best_val_f1 = 0
    best_model_state = None
    best_epoch = 0
    patience = config.PATIENCE
    patience_counter = 0
    fold_history = []
    model_save_path = Path(config.MODELS_DIR) / f"fold_{fold + 1}_best_model.pth"
    os.makedirs(model_save_path.parent, exist_ok=True)
    os.makedirs(config.HISTORY_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "fold_reports"), exist_ok=True)
    
    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{config.EPOCHS} ---")
        if config.UNFREEZE_EPOCH and (epoch + 1) >= config.UNFREEZE_EPOCH and not model.backbone_finetuned:
            model.enable_backbone_finetuning(config.UNFREEZE_ACOUSTIC_LAYERS, config.UNFREEZE_LINGUISTIC_LAYERS)
            print("Backbone fine-tuning enabled for Wav2Vec2/BERT last layers.")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        train_loss, train_acc, train_f1, train_prec, train_rec, train_auc = train_metrics
        print(
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | "
            f"Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | AUC: {train_auc:.4f}"
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc, _, _, _ = val_metrics
        print(
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
            f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | AUC: {val_auc:.4f}"
        )
        current_lr = optimizer.param_groups[0]['lr']
        fold_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "learning_rate": current_lr,
        })
        
        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # state_dict() holds live tensors that keep training; clone so the best weights are kept
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, model_save_path)
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"New best validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    if best_model_state is None:
        best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    test_loss, test_acc, test_f1, test_prec, test_rec, test_auc, test_preds, test_labels, _ = test_metrics
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    history_path = Path(config.HISTORY_DIR) / f"fold_{fold + 1}_history.csv"
    pd.DataFrame(fold_history).to_csv(history_path, index=False)
    report_path = Path(config.RESULTS_DIR) / "fold_reports" / f"fold_{fold + 1}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(test_labels, test_preds))
    
    print(f"\nFold {fold+1} Results:")
    print(
        f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | "
        f"Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | AUC: {test_auc:.4f}"
    )
    print(f"Classification report saved to {report_path}")
    
    return {
        'fold': fold + 1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'best_val_f1': best_val_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_auc': test_auc,
        'cm_tn': int(tn),
        'cm_fp': int(fp),
        'cm_fn': int(fn),
        'cm_tp': int(tp),
        'history_path': str(history_path),
        'model_path': str(model_save_path),
        'best_epoch': best_epoch,
    }

def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Check if metadata file exists
    if not os.path.exists(config.OUTPUT_CSV_PATH):
        print("Metadata file not found. Please run preprocess.py first.")
        return
    
    # Load processors and dataset
    acoustic_processor, linguistic_tokenizer = get_processors()
    full_dataset = EnhancedDementiaDataset(config.OUTPUT_CSV_PATH)
    
    labels = full_dataset.labels
    groups = full_dataset.groups
    print(f"Total dataset size: {len(full_dataset)} "
          f"({len(np.unique(groups))} speakers, {int(labels.sum())} impaired (1) / {int((labels == 0).sum())} control (0))")

    fold_results = []

    # Create indices array
    dataset_indices = np.arange(len(full_dataset))

    metadata = full_dataset.metadata
    if config.USE_OFFICIAL_SPLIT and 'split' in metadata.columns:
        # One run on the dataset's predefined partition (PROCESS-2), comparable with published baselines
        is_test = metadata['split'].astype(str).str.lower().eq('test').to_numpy()
        print(f"Using the dataset's official split: {int((~is_test).sum())} train+val, {int(is_test.sum())} test")
        splits = [(dataset_indices[~is_test], dataset_indices[is_test])]
    else:
        # 10-fold cross-validation. Stratified so every split contains both classes (the metadata
        # CSV lists all controls before all dementia, so unstratified slicing gave all-dementia
        # validation sets), and grouped so one speaker's recordings never span train and test.
        outer_cv = StratifiedGroupKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        splits = outer_cv.split(dataset_indices, labels, groups)

    for fold, (temp_indices, test_indices) in enumerate(splits):
        # Split temp_indices into 8 train + 1 validation, also stratified and speaker-grouped
        inner_cv = StratifiedGroupKFold(n_splits=config.K_FOLDS - 1, shuffle=True, random_state=config.RANDOM_SEED)
        inner_train, inner_val = next(inner_cv.split(temp_indices, labels[temp_indices], groups[temp_indices]))
        train_indices = temp_indices[inner_train]
        val_indices = temp_indices[inner_val]

        print(f"\nFold {fold+1}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} | "
              f"dementia share train/val/test = {labels[train_indices].mean():.2f}/"
              f"{labels[val_indices].mean():.2f}/{labels[test_indices].mean():.2f}")
        
        # Train this fold
        fold_result = train_fold(fold, train_indices, val_indices, test_indices,
                                full_dataset, acoustic_processor, linguistic_tokenizer, device)
        fold_results.append(fold_result)
    
    # Calculate and display overall results
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS" if len(fold_results) > 1 else "OFFICIAL TEST SPLIT RESULTS")
    print(f"{'='*50}")
    
    accuracies = [result['test_accuracy'] for result in fold_results]
    f1_scores = [result['test_f1'] for result in fold_results]
    
    print(f"Mean Test Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean Test F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    print("\nPer-fold results:")
    for result in fold_results:
        print(f"Fold {result['fold']:2d}: Acc={result['test_accuracy']:.4f}, F1={result['test_f1']:.4f}")
    
    # Save results to file
    results_df = pd.DataFrame(fold_results)
    results_path = Path(config.RESULTS_DIR) / "10fold_results.csv"
    os.makedirs(results_path.parent, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()