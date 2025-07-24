import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchaudio.transforms as T

class VoiceDataset(Dataset):
    """
    Dataset for loading spectrograms with on-the-fly training augmentations.
    """
    def __init__(self, csv_path, mean, std, is_train=True):
        self.data = pd.read_csv(csv_path)
        mels_flat = np.array([np.fromstring(mel, sep=' ') for mel in self.data['mel_flat']], dtype=np.float32)
        self.mels = mels_flat.reshape(-1, 1, 80, 50) # Reshape to (N, C, H, W)
        
        self.mean = mean
        self.std = std
        self.is_train = is_train

        # Define augmentation pipeline for training data
        if self.is_train:
            self.augmentation = torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=15), # Mask out frequency bands
                T.TimeMasking(time_mask_param=35)      # Mask out time steps
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Already in (C, H, W) format
        mel = torch.from_numpy(self.mels[index]).float()
        
        # Apply augmentations only during training
        if self.is_train:
            mel = self.augmentation(mel)
            # Add a small amount of Gaussian noise
            noise = torch.randn_like(mel) * 0.05 
            mel += noise

        # Apply normalization
        mel_norm = (mel - self.mean) / self.std
        
        return mel_norm

def get_autoencoder_dataloader(csv_path, stats_path, batch_size, shuffle=True, is_train=True, num_workers=2):
    """
    Creates a DataLoader. Applies augmentations if is_train is True.
    """
    try:
        stats = np.load(stats_path)
        mean, std = stats['mean'], stats['std']
    except FileNotFoundError:
        raise RuntimeError(f"Normalization file '{stats_path}' not found.")

    dataset = VoiceDataset(csv_path, mean=mean, std=std, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
