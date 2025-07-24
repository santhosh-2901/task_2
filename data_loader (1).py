import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random

class TripletVoiceDataset(Dataset):
    """
    Creates triplets of (anchor, positive, negative) for training with TripletMarginLoss.
    """
    def __init__(self, csv_path, mean, std):
        self.data = pd.read_csv(csv_path)
        # Create a mapping from each speaker to the list of their sample indices
        self.speaker_to_indices = {speaker: np.where(self.data.speaker_id == speaker)[0]
                                   for speaker in self.data.speaker_id.unique()}
        
        self.mels = np.array([np.fromstring(mel, sep=' ') for mel in self.data['mel_flat']])
        self.mels = self.mels.reshape(-1, 1, 80, 50) # Reshape for Conv2d
        
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # --- Anchor ---
        anchor_mel = self.mels[index]
        anchor_speaker = self.data.iloc[index].speaker_id
        
        # --- Positive (another sample from the same speaker) ---
        positive_indices = self.speaker_to_indices[anchor_speaker]
        # Ensure the positive sample is different from the anchor
        positive_index = random.choice([i for i in positive_indices if i != index])
        positive_mel = self.mels[positive_index]

        # --- Negative (a sample from a different speaker) ---
        negative_speakers = [s for s in self.speaker_to_indices if s != anchor_speaker]
        negative_speaker = random.choice(negative_speakers)
        negative_index = random.choice(self.speaker_to_indices[negative_speaker])
        negative_mel = self.mels[negative_index]

        # Normalize all three spectrograms
        anchor_norm = (anchor_mel - self.mean) / self.std
        positive_norm = (positive_mel - self.mean) / self.std
        negative_norm = (negative_mel - self.mean) / self.std
        
        return {
            'anchor': torch.from_numpy(anchor_norm).float(),
            'positive': torch.from_numpy(positive_norm).float(),
            'negative': torch.from_numpy(negative_norm).float(),
        }

def get_triplet_dataloader(csv_path, batch_size, shuffle=True):
    """ Creates a DataLoader for the Triplet network. """
    if 'train' in os.path.basename(csv_path).lower():
        print("Computing normalization stats from training data...")
        full_data = pd.read_csv(csv_path)
        all_mels = np.array([np.fromstring(mel, sep=' ') for mel in full_data['mel_flat']])
        mean = all_mels.mean().astype(np.float32)
        std = all_mels.std().astype(np.float32)
        np.savez('normalization_stats_triplet.npz', mean=mean, std=std)
        print(f"Stats computed and saved (mean={mean:.4f}, std={std:.4f}).")
        dataset = TripletVoiceDataset(csv_path, mean=mean, std=std)
    else:
        stats = np.load('normalization_stats_triplet.npz')
        mean, std = stats['mean'], stats['std']
        print(f"Loaded normalization stats (mean={mean:.4f}, std={std:.4f}).")
        dataset = TripletVoiceDataset(csv_path, mean=mean, std=std)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)