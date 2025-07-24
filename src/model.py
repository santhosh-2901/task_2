import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder with Dropout for regularization.
    """
    def __init__(self, input_channels=1, embedding_dim=128, dropout_rate=0.4):
        super(SpeakerEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 10 * 6, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return F.normalize(x, p=2, dim=1)

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    This helps in training deeper networks by allowing gradients to flow more easily.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The core of the residual connection
        out = F.leaky_relu(out, 0.2)
        return out

class MelDecoder(nn.Module):
    """
    A state-of-the-art Mel-Spectrogram Decoder using residual blocks for enhanced capacity.
    """
    def __init__(self, embedding_dim=128, output_channels=1, dropout_rate=0.4):
        super(MelDecoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128 * 10 * 6)
        )

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # -> (B, 64, 20, 12)
        self.res1 = ResidualBlock(64, 64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> (B, 32, 40, 24)
        self.res2 = ResidualBlock(32, 32)
        
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 6), stride=(2, 2), padding=(1, 1)) # -> (B, 16, 80, 50)
        self.res3 = ResidualBlock(16, 16)

        # Final convolution to produce the single-channel mel-spectrogram.
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), 128, 10, 6)
        
        x = self.deconv1(x)
        x = self.res1(x)
        
        x = self.deconv2(x)
        x = self.res2(x)
        
        x = self.deconv3(x)
        x = self.res3(x)

        x = self.final_conv(x)
        return x

class VoiceCloningModel(nn.Module):
    """
    The complete autoencoder-style model for voice cloning.
    """
    def __init__(self, embedding_dim=128, dropout_rate=0.4):
        super(VoiceCloningModel, self).__init__()
        self.speaker_encoder = SpeakerEncoder(embedding_dim=embedding_dim, dropout_rate=dropout_rate)
        self.mel_decoder = MelDecoder(embedding_dim=embedding_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        speaker_embedding = self.speaker_encoder(x)
        reconstructed_mel = self.mel_decoder(speaker_embedding)
        return reconstructed_mel, speaker_embedding

    def clone_voice(self, reference_mel):
        if reference_mel.dim() == 3:
            reference_mel = reference_mel.unsqueeze(1)
        speaker_embedding = self.speaker_encoder(reference_mel)
        cloned_mel = self.mel_decoder(speaker_embedding)
        return cloned_mel
