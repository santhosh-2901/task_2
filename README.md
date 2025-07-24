
## How to Run

### 1. Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt

Ensure your data files (mel_train.csv, mel_val.csv, mel_reference.csv) are placed in a data/ directory or update the paths in the scripts.

2. Step 1: Train the Speaker Encoder
This step trains the encoder to create meaningful speaker embeddings using triplet loss.

python train.py

This will produce speaker_encoder_triplet_best.pth and normalization_stats_triplet.npz.

3. Step 2: Train the Mel Decoder
This step freezes the encoder and trains the decoder to reconstruct mel-spectrograms.

python train_autoencoder.py

This will create voice_cloning_model_best.pth, normalization_stats.npz, and a training_curve.png plot.

4. Step 3: Run Inference
Finally, generate the cloned mel-spectrograms for the reference utterances.

python infer.py

This will generate the final deliverable: cloned_mel_predictions.csv.

Requirements
torch>=1.12
pandas
numpy
matplotlib
