import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Note on F0_compressed:
# A nn.Module's forward() pass must operate on Tensors for batching and gradients.
# The 'F0_compressed' dictionary must be *reconstructed* into a tensor *before*
# being passed to this module. The `forward` signature is therefore:
# forward(self, z_q, e_a, f0_recon)
# where f0_recon is the reconstructed pitch contour tensor.

class ResBlock(nn.Module):
    """
    HiFi-GAN Residual Block with dilations.
    """
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            # Calculate padding to keep time dimension the same
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 
                                      dilation=d, padding=padding))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 
                                      dilation=1, padding=(kernel_size - 1) // 2))
            )
        
        # Initialize weights
        self.convs1.apply(self.init_weights)
        self.convs2.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt  # Residual connection
        return x

class MRFBlock(nn.Module):
    """
    HiFi-GAN Multi-Receptive Field (MRF) Block.
    """
    def __init__(self, channels, kernel_sizes, dilations_list):
        super().__init__()
        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations_list):
            self.blocks.append(ResBlock(channels, k, d))
    
    def forward(self, x):
        # Sum the outputs of all parallel residual blocks
        out = torch.zeros_like(x)
        for block in self.blocks:
            out += block(x)
        return out


class DecoderVocoder(nn.Module):
    """
    HiFi-GAN–style Generator (Decoder).
    
    Reconstructs a waveform from aligned latent and conditioning signals.
    """
    def __init__(self,
                 z_dim: int,
                 e_a_dim: int,
                 hidden_dim: int = 512,
                 upsample_factors: tuple = (8, 8, 2, 2),
                 mrf_kernels: list = [3, 7, 11],
                 mrf_dilations: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
                 ):
        super().__init__()
        
        # 1. Conditioning input projection
        # Input: z_q (z_dim) + e_a (e_a_dim) + f0 (1)
        self.in_channels = z_dim + e_a_dim + 1
        self.pre_conv = weight_norm(nn.Conv1d(self.in_channels, hidden_dim, 
                                              kernel_size=7, padding=3))
        
        self.upsample_stack = nn.ModuleList()
        current_dim = hidden_dim
        
        # 2. Upsampling stack (ConvTranspose1d + MRF)
        for i, factor in enumerate(upsample_factors):
            self.upsample_stack.append(
                weight_norm(nn.ConvTranspose1d(
                    current_dim,
                    current_dim // 2,
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2 + (factor % 2)
                ))
            )
            self.upsample_stack.append(
                MRFBlock(current_dim // 2, mrf_kernels, mrf_dilations)
            )
            current_dim //= 2
            
        # 3. Final projection to 1-channel (audio waveform)
        self.post_conv = weight_norm(nn.Conv1d(current_dim, 1, 
                                               kernel_size=7, padding=3))
        
        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, z_q, e_a, f0_recon):
        """
        Runs the decoder/vocoder.
        
        Args:
            z_q (torch.Tensor): Quantized main latents (B, z_dim, T_z)
                                (e.g., T_z = T_mel // 8)
            e_a (torch.Tensor): Quantized accent embedding (B, e_a_dim, T_a)
                                (e.g., T_a = T_mel // 12)
            f0_recon (torch.Tensor): Reconstructed F0 contour (B, T_mel)
            
        Returns:
            torch.Tensor: Reconstructed waveform (B, 1, T_wav)
        """
        
        # 1. Align all inputs to mel-spectrogram time resolution (T_mel)
        T_mel = f0_recon.shape[-1]
        
        # Upsample z_q (B, z_dim, T_z) -> (B, z_dim, T_mel)
        z_q_up = F.interpolate(z_q, size=T_mel, mode='nearest')
        
        # Upsample e_a (B, e_a_dim, T_a) -> (B, e_a_dim, T_mel)
        e_a_up = F.interpolate(e_a, size=T_mel, mode='nearest')
        
        # Add channel dim to f0 (B, T_mel) -> (B, 1, T_mel)
        f0_up = f0_recon.unsqueeze(1)
        
        # 2. Concatenate all conditioning signals
        x = torch.cat([z_q_up, e_a_up, f0_up], dim=1)
        
        # 3. Run through the HiFi-GAN generator stack
        x = self.pre_conv(x)
        
        for layer in self.upsample_stack:
            if isinstance(layer, nn.ConvTranspose1d):
                x = F.leaky_relu(x, 0.1)
            x = layer(x)
            
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        
        # 4. Final Tanh activation to output a waveform in [-1, 1] range
        y_hat = torch.tanh(x)
        
        return y_hat


# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Config ---
    Z_DIM = 128       # From encoder.py
    E_A_DIM = 128     # From accent_encoder.py
    HOP_LENGTH = 256  # From feature_extraction.py
    
    # --- Dummy Inputs ---
    BATCH_SIZE = 4
    T_MEL = 100       # 100 mel frames
    T_WAV = T_MEL * HOP_LENGTH # 100 * 256 = 25600 samples
    
    # Create inputs with different time resolutions
    T_z = 13          # 100 // 8 = 12, but Conv padding might add one
    T_a = 9           # 100 // 12 = 8, but Conv padding might add one
    
    dummy_z_q = torch.randn(BATCH_SIZE, Z_DIM, T_z)
    dummy_e_a = torch.randn(BATCH_SIZE, E_A_DIM, T_a)
    dummy_f0_recon = torch.abs(torch.randn(BATCH_SIZE, T_MEL)) # F0 is positive
    
    print("--- DecoderVocoder Test ---")
    print(f"Target audio length: {T_WAV} samples")
    
    print(f"\nInput z_q shape:    {list(dummy_z_q.shape)}")
    print(f"Input e_a shape:    {list(dummy_e_a.shape)}")
    print(f"Input f0_recon shape: {list(dummy_f0_recon.shape)}")
    
    # --- Initialize and run model ---
    decoder = DecoderVocoder(z_dim=Z_DIM, e_a_dim=E_A_DIM)
    
    y_hat = decoder(dummy_z_q, dummy_e_a, dummy_f0_recon)
    
    print(f"\nOutput y_hat shape:   {list(y_hat.shape)}")
    
    # Check output length
    output_len = y_hat.shape[-1]
    print(f"Output vs Target length: {output_len} vs {T_WAV}")
    assert output_len == T_WAV, "Output length does not match expected T_mel * HOP_LENGTH"
    print("\n✅ Test passed!")