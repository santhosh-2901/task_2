import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import get_triplet_dataloader
from model import SpeakerEncoder

# --- Configuration ---
TRAIN_CSV = '/content/data/mel_train.csv'
VAL_CSV = '/content/data/mel_val.csv'
MODEL_SAVE_PATH = 'speaker_encoder_triplet_best.pth'
NUM_EPOCHS = 100 # Triplet loss often converges faster
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    """ Main training loop for the Speaker Encoder with Triplet Loss """
    train_loader = get_triplet_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_triplet_dataloader(VAL_CSV, batch_size=BATCH_SIZE, shuffle=False)

    model = SpeakerEncoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # The margin is a critical hyperparameter. 1.0 is a good starting point.
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    best_val_loss = float('inf')
    print(f"Starting Triplet Loss training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            anchor = batch['anchor'].to(DEVICE)
            positive = batch['positive'].to(DEVICE)
            negative = batch['negative'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Get embeddings for all three samples
            output_anchor = model(anchor)
            output_positive = model(positive)
            output_negative = model(negative)
            
            loss = criterion(output_anchor, output_positive, output_negative)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                anchor = batch['anchor'].to(DEVICE)
                positive = batch['positive'].to(DEVICE)
                negative = batch['negative'].to(DEVICE)
                
                output_anchor = model(anchor)
                output_positive = model(positive)
                output_negative = model(negative)
                
                loss = criterion(output_anchor, output_positive, output_negative)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" ⭐ New best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    train()