import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from data_loader_autoencoder import get_autoencoder_dataloader
from model import VoiceCloningModel

# --- Configuration ---
TRAIN_CSV = 'mel_train.csv'
VAL_CSV = 'mel_val.csv'
PRETRAINED_ENCODER_PATH = 'speaker_encoder_triplet_best.pth'
STATS_PATH = 'normalization_stats_triplet.npz'
MODEL_SAVE_PATH = 'voice_cloning_model_best.pth'
NUM_EPOCHS = 200
BATCH_SIZE = 32

# --- Regularization & Optimization Hyperparameters ---
BASE_DECODER_LR = 2e-4 # Slightly higher LR for the more powerful ResNet decoder
BASE_ENCODER_LR = 5e-6 
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.4
WARMUP_EPOCHS = 10 
SCHEDULER_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def reconstruction_loss(y_true, y_pred):
    """
    Composite loss using L1 (MAE) for sharper spectrograms, plus Cosine Similarity.
    """
    l1_loss = nn.L1Loss()(y_true, y_pred)
    cos_loss = 1 - nn.functional.cosine_similarity(y_true.flatten(1), y_pred.flatten(1)).mean()
    return l1_loss + cos_loss

def adjust_learning_rate(optimizer, epoch, base_lrs, warmup_epochs):
    """Manually adjusts learning rate with a warm-up schedule."""
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = base_lrs[i] * warmup_factor

def train():
    train_loader = get_autoencoder_dataloader(TRAIN_CSV, STATS_PATH, batch_size=BATCH_SIZE, shuffle=True, is_train=True)
    val_loader = get_autoencoder_dataloader(VAL_CSV, STATS_PATH, batch_size=BATCH_SIZE, shuffle=False, is_train=False)

    model = VoiceCloningModel(dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    if os.path.exists(PRETRAINED_ENCODER_PATH):
        print(f"Loading pre-trained speaker encoder from {PRETRAINED_ENCODER_PATH}")
        encoder_state_dict = torch.load(PRETRAINED_ENCODER_PATH, map_location=DEVICE)
        model.speaker_encoder.load_state_dict(encoder_state_dict, strict=False)
    else:
        print("Warning: No pre-trained encoder found. Training entire model from scratch.")

    optimizer = optim.Adam([
        {'params': model.speaker_encoder.parameters(), 'lr': BASE_ENCODER_LR},
        {'params': model.mel_decoder.parameters(), 'lr': BASE_DECODER_LR}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.2, verbose=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=MODEL_SAVE_PATH)

    train_losses, val_losses = [], []
    print(f"Starting unified training on {DEVICE} with ResNet Decoder and L1 Loss...")

    for epoch in range(NUM_EPOCHS):
        adjust_learning_rate(optimizer, epoch, [BASE_ENCODER_LR, BASE_DECODER_LR], WARMUP_EPOCHS)
        current_lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Current LRs: Encoder={current_lrs[0]:.2e}, Decoder={current_lrs[1]:.2e}")

        model.train()
        total_train_loss = 0
        for batch in train_loader:
            mels = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstructed_mels, _ = model(mels)
            loss = reconstruction_loss(mels, reconstructed_mels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                mels = batch.to(DEVICE)
                reconstructed_mels, _ = model(mels)
                loss = reconstruction_loss(mels, reconstructed_mels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step(avg_val_loss)
            
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH} with validation loss: {early_stopping.val_loss_min:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=WARMUP_EPOCHS -1, color='orange', linestyle='--', label='End of Warm-up')
    if early_stopping.early_stop:
        stop_epoch = len(val_losses) - early_stopping.patience - 1
        plt.axvline(x=stop_epoch, color='r', linestyle='--', label='Early Stopping Point')
    plt.title('Training and Validation Loss Curve with ResNet Decoder')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    print("Training curve saved to training_curve.png")

if __name__ == '__main__':
    train()
