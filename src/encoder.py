import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    CNN-based Encoder to downsample mel-spectrograms into a latent tensor z.
    
    Input shape: (batch_size, n_mels, n_frames)
    Output shape: (batch_size, latent_dim, n_frames // downsample_factor)
    """
    def __init__(self, in_channels=80, latent_dim=128, channels=512):
        super().__init__()
        
        # We expect input as (B, N_MELS, T). 
        # Conv1d expects (B, C_in, T), so N_MELS (80) is our in_channels.
        # This architecture uses 1D convolutions over the time dimension.
        
        self.net = nn.Sequential(
            # Input: (B, 80, T)
            nn.Conv1d(in_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # Downsample 1
            nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # Shape: (B, 512, T/2)

            # Downsample 2
            nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # Shape: (B, 512, T/4)

            # Downsample 3
            nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # Shape: (B, 512, T/8)
            
            # Final projection to latent dimension
            nn.Conv1d(channels, latent_dim, kernel_size=3, stride=1, padding=1)
            # Output: (B, 128, T/8)
        )
        # The total downsampling factor is 2*2*2 = 8

    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram (torch.Tensor): Input tensor of shape (B, n_mels, n_frames).
                                            n_mels should match in_channels (e.g., 80).
        Returns:
            torch.Tensor: Latent tensor z of shape (B, latent_dim, n_frames // 8).
        """
        # (B, 80, T) -> (B, 128, T/8)
        z = self.net(mel_spectrogram)
        return z

# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    N_MELS = 80       # From feature_extraction.py
    LATENT_DIM = 128  # Desired latent dimension
    
    # Create a dummy input tensor
    # B: Batch size (e.g., 4)
    # N_MELS: Mel bands (80)
    # T: Number of frames (e.g., 256, must be divisible by 8 for this example)
    dummy_mel = torch.randn(4, N_MELS, 256)
    
    # Initialize the encoder
    encoder = Encoder(in_channels=N_MELS, latent_dim=LATENT_DIM)
    
    # Pass the dummy input through the encoder
    z = encoder(dummy_mel)
    
    print(f"--- Encoder Test ---")
    print(f"Input mel shape:  {list(dummy_mel.shape)}")
    print(f"Output latent z shape: {list(z.shape)}")
    print(f"Downsample factor: {dummy_mel.shape[2] // z.shape[2]}")
    print(f"Latent dimension:  {z.shape[1]}")

    # --- Test with a more realistic frame length ---
    dummy_mel_2 = torch.randn(4, N_MELS, 301) # 301 is not divisible by 8
    z_2 = encoder(dummy_mel_2)
    print(f"\nInput mel shape:  {list(dummy_mel_2.shape)}")
    print(f"Output latent z shape: {list(z_2.shape)}") # Should be (4, 128, 37)