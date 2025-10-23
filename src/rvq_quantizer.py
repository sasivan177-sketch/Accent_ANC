import torch
import torch.nn as nn
from typing import Dict

# --- Configuration ---

# We calculate the frames per second from src/feature_extraction.py
# TARGET_SR = 16000, HOP_LENGTH = 256
FRAMES_PER_SEC = 16000 / 256  # 62.5 frames/sec

# We use 10 bits per quantizer (codebook_size = 2^10 = 1024)
BITS_PER_QUANTIZER = 10
CODEBOOK_SIZE = 1024

def bits_per_frame(kbps: int) -> int:
    """Calculates target bits per frame for a given bitrate in kbps."""
    return (kbps * 1000) / FRAMES_PER_SEC # e.g., (16 * 1000) / 62.5 = 256 bits/frame

def num_quantizers_for_bitrate(kbps: int) -> int:
    """Calculates how many quantizers to use for a given bitrate."""
    # We use ceil to ensure we meet or exceed the target bits
    # e.g., 2 kbps -> 32 bits/frame. 32 / 10 = 3.2. We use 4 quantizers.
    # Let's adjust to use floor to stay *under* the bitrate, which is more common.
    # 2 kbps -> 32 bits -> 3 quantizers (30 bits)
    # 4 kbps -> 64 bits -> 6 quantizers (60 bits)
    # 8 kbps -> 128 bits -> 12 quantizers (120 bits)
    # 16 kbps -> 256 bits -> 25 quantizers (250 bits)
    return int(bits_per_frame(kbps) // BITS_PER_QUANTIZER)


class QuantizationLayer(nn.Module):
    """
    A single Vector Quantization (VQ) layer.
    Takes a continuous vector and maps it to the closest entry in a codebook.
    """
    def __init__(self, in_channels: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, in_channels)
        
    def forward(self, z: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            z (torch.Tensor): Input tensor of shape (B, C, T)

        Returns:
            torch.Tensor: Quantized tensor z_q of shape (B, C, T)
            torch.Tensor: Codebook indices of shape (B, T)
        """
        # (B, C, T) -> (B, T, C)
        z_transposed = z.transpose(1, 2)
        # (B, T, C) -> (B*T, C)
        z_flat = z_transposed.contiguous().view(-1, z.size(1))

        # Find the closest codebook vectors (efficient L2 distance calculation)
        # (z_flat^2).sum(1) -> (B*T, 1)
        # self.codebook.weight.T -> (C, codebook_size)
        # (z_flat @ self.codebook.weight.T) -> (B*T, codebook_size)
        # (self.codebook.weight^2).sum(1) -> (codebook_size, 1)
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * (z_flat @ self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        
        # Find the indices of the minimum distances
        # indices shape: (B*T)
        indices = distances.argmin(dim=1)
        
        # Retrieve the quantized vectors from the codebook
        # z_q_flat shape: (B*T, C)
        z_q_flat = self.codebook(indices)

        # Reshape back to (B, C, T)
        # z_q = z_q_flat.view(z_transposed.shape).transpose(1, 2)
        z_q = z_q_flat.view_as(z_transposed).transpose(1, 2).contiguous()
        
        # Reshape indices to (B, T)
        indices_reshaped = indices.view(z.size(0), z.size(2))

        return z_q, indices_reshaped


class RVQQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).
    Applies multiple stages of quantization, each on the residual of the last.
    """
    def __init__(self, in_channels: int, codebook_size: int = CODEBOOK_SIZE):
        super().__init__()
        
        self.bitrate_map: Dict[int, int] = {
            kbps: num_quantizers_for_bitrate(kbps) 
            for kbps in [2, 4, 8, 16]
        }
        
        self.max_quantizers = self.bitrate_map[16] # e.g., 25
        
        # Create a list of quantizer layers
        self.quantizers = nn.ModuleList([
            QuantizationLayer(in_channels, codebook_size)
            for _ in range(self.max_quantizers)
        ])

    def forward(self, z: torch.Tensor, bitrate_kbps: int = 16) -> (torch.Tensor, torch.Tensor):
        """
        Quantizes z and returns the reconstructed vector and the codes.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, T).
            bitrate_kbps (int): Target bitrate. Must be one of {2, 4, 8, 16}.
            
        Returns:
            torch.Tensor: Reconstructed z_q of shape (B, C, T).
            torch.Tensor: Stacked indices of shape (num_quantizers, B, T).
        """
        if bitrate_kbps not in self.bitrate_map:
            raise ValueError(f"Bitrate {bitrate_kbps}kbps not supported. Must be one of {list(self.bitrate_map.keys())}")

        num_quantizers_to_use = self.bitrate_map[bitrate_kbps]
        
        residual = z
        all_quantized_vectors = []
        all_indices = []

        for i in range(num_quantizers_to_use):
            quantizer = self.quantizers[i]
            
            # Quantize the current residual
            z_q_i, indices_i = quantizer(residual)
            
            # Add the quantized vector to our list
            all_quantized_vectors.append(z_q_i)
            
            # Add the indices to our list
            all_indices.append(indices_i)
            
            # Update the residual for the next stage
            residual = residual - z_q_i

        # The final reconstructed vector is the sum of all quantized vectors
        z_q = torch.stack(all_quantized_vectors).sum(dim=0)
        
        # The final codes are the stack of all indices
        # Shape: (num_quantizers, B, T)
        indices = torch.stack(all_indices, dim=0)

        return z_q, indices
    
    def get_bitrate_map(self):
        print("--- Bitrate Configuration ---")
        print(f"Frames per second: {FRAMES_PER_SEC:.2f}")
        print(f"Bits per quantizer: {BITS_PER_QUANTIZER} (Codebook size: {CODEBOOK_SIZE})")
        print("Bitrate -> Quantizers -> Actual Bitrate:")
        for kbps, n_quant in self.bitrate_map.items():
            bits_per_frame = n_quant * BITS_PER_QUANTIZER
            actual_kbps = (bits_per_frame * FRAMES_PER_SEC) / 1000
            print(f"  {kbps:2d} kbps -> {n_quant:2d} quantizers -> {bits_per_frame:3d} bits/frame ({actual_kbps:.2f} kbps)")


# --- Example Usage ---
if __name__ == "__main__":
    LATENT_DIM = 128  # Must match Encoder's latent_dim
    BATCH_SIZE = 4
    FRAMES = 37       # Example: 301 audio frames -> 37 latent frames (8x downsample)
    
    # Create a dummy latent tensor z (output from Encoder)
    dummy_z = torch.randn(BATCH_SIZE, LATENT_DIM, FRAMES)
    
    # Initialize the RVQ
    rvq = RVQQuantizer(in_channels=LATENT_DIM)
    
    rvq.get_bitrate_map()

    # --- Test 16 kbps mode ---
    z_q_16, indices_16 = rvq(dummy_z, bitrate_kbps=16)
    
    print("\n--- 16 kbps Test ---")
    print(f"Input z shape:     {list(dummy_z.shape)}")
    print(f"Quantized z_q shape: {list(z_q_16.shape)}")
    print(f"Indices shape:     {list(indices_16.shape)}  (Quantizers, Batch, Frames)")

    # --- Test 2 kbps mode ---
    z_q_2, indices_2 = rvq(dummy_z, bitrate_kbps=2)
    
    print("\n--- 2 kbps Test ---")
    print(f"Input z shape:     {list(dummy_z.shape)}")
    print(f"Quantized z_q shape: {list(z_q_2.shape)}")
    print(f"Indices shape:     {list(indices_2.shape)}  (Quantizers, Batch, Frames)")