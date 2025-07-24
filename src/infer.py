import torch
import pandas as pd
import numpy as np
import os

from model import VoiceCloningModel

# --- Configuration ---
REFERENCE_CSV = '/content/data/mel_reference.csv'
MODEL_PATH = 'voice_cloning_model_best.pth'
OUTPUT_CSV = 'cloned_mel_predictions.csv'
# This must be the same stats file used during training
STATS_PATH = 'normalization_stats_triplet.npz' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer():
    """
    Runs inference on the reference utterances to generate cloned mel-spectrograms.
    """
    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure training is complete.")
    
    print(f"Loading model from {MODEL_PATH}")
    # Use the same dropout rate as in training for consistency, although it's in eval mode
    model = VoiceCloningModel(dropout_rate=0.4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- Load Normalization Stats ---
    try:
        stats = np.load(STATS_PATH)
        mean, std = stats['mean'], stats['std']
        print(f"Loaded normalization stats (mean={mean:.4f}, std={std:.4f}).")
    except FileNotFoundError:
        raise RuntimeError(f"Normalization file '{STATS_PATH}' not found. Cannot run inference without it.")

    # --- Load Reference Data ---
    ref_df = pd.read_csv(REFERENCE_CSV)
    print(f"Loaded {len(ref_df)} reference utterances.")

    predictions = []
    with torch.no_grad():
        for index, row in ref_df.iterrows():
            speaker_id = row['speaker_id']
            mel_flat = np.fromstring(row['mel_flat'], sep=' ', dtype=np.float32)
            
            # Reshape and normalize the reference mel
            mel = mel_flat.reshape(1, 80, 50) # (C, H, W)
            mel_norm = (mel - mean) / std
            
            # Convert to tensor and add batch dimension
            ref_tensor = torch.from_numpy(mel_norm).float().unsqueeze(0).to(DEVICE) # (B, C, H, W)
            
            # --- Perform Voice Cloning ---
            cloned_mel_norm = model.clone_voice(ref_tensor)
            
            # Denormalize the output
            cloned_mel_tensor = cloned_mel_norm.squeeze().cpu().numpy() * std + mean
            
            # Flatten for saving
            predicted_mel_flat = ' '.join(map(str, cloned_mel_tensor.flatten()))
            
            predictions.append({
                'speaker_id': speaker_id,
                'predicted_mel_flat': predicted_mel_flat
            })
            print(f"Cloned voice for speaker {speaker_id}.")

    # --- Save Predictions ---
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved {len(output_df)} predictions to {OUTPUT_CSV}")

if __name__ == '__main__':
    infer()
