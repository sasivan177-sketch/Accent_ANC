import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.fft # Used for offline file processing only
from pathlib import Path
from tqdm import tqdm
import warnings

# --- Configuration ---
FEATURES_DIR = Path("data/features")
COMPRESSED_DIR = Path("data/compressed/pitch_codes")

# This is the key hyperparameter.
# Any DCT coefficient with an absolute value less than this
# will be set to 0. A larger value means more compression.
DCT_THRESHOLD = 0.1

# --- Utility Functions (For OFFLINE Data Pre-compression - NOT Differentiable) ---

def compress_f0(f0: np.ndarray) -> dict:
    """
    [OFFLINE UTILITY] Compresses an F0 contour using NumPy/SciPy DCT on the residual.
    This is for saving the compressed data, not for end-to-end training.
    """
    unvoiced_flags = (f0 == 0)
    voiced_f0 = f0[~unvoiced_flags]

    if len(voiced_f0) == 0:
        mean_f0 = 0.0
        f0_residual = f0
    else:
        # 1. Calculate the mean and the residual
        mean_f0 = np.mean(voiced_f0)
        f0_residual = f0.copy()
        f0_residual[~unvoiced_flags] -= mean_f0

    # 2. Apply Discrete Cosine Transform (DCT)
    f0_dct = scipy.fft.dct(f0_residual, type=2, norm='ortho')

    # 3. Apply threshold to get sparse coefficients
    f0_dct_sparse = f0_dct.copy()
    # The DCT_THRESHOLD here controls the saved bitstream size
    f0_dct_sparse[np.abs(f0_dct_sparse) < DCT_THRESHOLD] = 0.0
    
    # 4. Store the data needed for reconstruction
    compressed_data = {
        'sparse_dct_coeffs': f0_dct_sparse.astype(np.float32),
        'mean_f0': float(mean_f0),
        'unvoiced_flags': unvoiced_flags
    }
    return compressed_data

def reconstruct_f0(compressed_data: dict) -> np.ndarray:
    """
    [OFFLINE UTILITY] Reconstructs the F0 contour from the compressed data.
    """
    sparse_coeffs = compressed_data['sparse_dct_coeffs']
    mean_f0 = compressed_data['mean_f0']
    unvoiced_flags = compressed_data['unvoiced_flags']
    
    # 1. Apply Inverse DCT (IDCT)
    f0_residual_recon = scipy.fft.idct(sparse_coeffs, type=2, norm='ortho')
    
    # 2. Add the mean back to the voiced parts
    f0_recon = f0_residual_recon
    f0_recon[~unvoiced_flags] += mean_f0
    
    # 3. Re-apply unvoiced flags (zeros) and clip
    f0_recon[unvoiced_flags] = 0.0
    f0_recon[f0_recon < 0] = 0.0
    
    return f0_recon


# --- ⭐️ PyTorch Module for End-to-End Training ⭐️ ---

class PitchResidualCoder(nn.Module):
    """
    Differentiable wrapper for Pitch Residual Coding (F0 compression/reconstruction).
    This simulates the lossy process using PyTorch ops for training L_Pitch.
    """
    def __init__(self, dct_threshold=DCT_THRESHOLD):
        super().__init__()
        # DCT Threshold is used here only as a reference; it's hard to make thresholding differentiable.
        self.dct_threshold = dct_threshold
        
    def forward(self, f0_original: torch.Tensor) -> torch.Tensor:
        """
        Performs differentiable pitch coding and reconstruction.
        
        Args:
            f0_original (torch.Tensor): The raw F0 contour (B, T_mel).
            
        Returns:
            torch.Tensor: The reconstructed F0 contour (B, T_mel).
        """
        B, T = f0_original.shape
        device = f0_original.device
        
        # 1. Separate voiced/unvoiced regions
        unvoiced_mask = (f0_original == 0).float() # 1 where unvoiced, 0 where voiced
        
        # 2. Mean Removal (Only for voiced parts)
        # We need to find the mean of the voiced values in a batch-wise differentiable way.
        voiced_count = (1.0 - unvoiced_mask).sum(dim=1, keepdim=True)
        # Prevent division by zero if entire batch element is unvoiced
        voiced_count[voiced_count == 0] = 1.0 
        
        # Calculate sum of voiced F0
        voiced_sum = (f0_original * (1.0 - unvoiced_mask)).sum(dim=1, keepdim=True)
        mean_f0 = voiced_sum / voiced_count
        
        # Calculate residual (subtract mean only from voiced parts)
        f0_residual = f0_original - mean_f0 * (1.0 - unvoiced_mask)
        
        # 3. DCT (Approximation using FFT) - Note: torch lacks native Type-II DCT
        # For end-to-end training, a common differentiable approximation or 
        # a simple low-pass filter is often used to simulate lossy compression.
        
        # --- Differentiable Low-Pass Filter (Proxy for Lossy DCT) ---
        
        # Simple smoothing kernel (Gaussian/Hann is better, but simple mean works)
        kernel_size = 5
        # The convolution expects (B, C, T) input
        f0_residual_in = f0_residual.unsqueeze(1) 
        
        # Create a simple 1D average filter
        kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        
        # Convolve to smooth/blur the residual (simulating lossy high-freq removal)
        f0_residual_recon = F.conv1d(f0_residual_in, kernel, padding=kernel_size // 2).squeeze(1)
        
        # 4. Add the mean back
        f0_recon = f0_residual_recon + mean_f0 * (1.0 - unvoiced_mask)

        # 5. Re-apply unvoiced regions and clip
        # Re-zero the unvoiced parts, as smoothing contaminates them
        f0_recon = f0_recon * (1.0 - unvoiced_mask)
        f0_recon[f0_recon < 0] = 0.0 # Clip non-physical values
        
        return f0_recon


# --- Main Runner ---

def main():
    """
    Finds all feature .npy files, compresses their F0,
    and saves the compressed data (using the OFFLINE numpy utility).
    """
    print("Starting F0 (pitch) compression...")
    # ... (rest of main function body is unchanged) ...
    print(f"  Features source: {FEATURES_DIR}")
    print(f"  Compressed data destination: {COMPRESSED_DIR}")
    print(f"  DCT Threshold: {DCT_THRESHOLD}")

    # Find all .npy files in the features directory
    feature_files = list(FEATURES_DIR.rglob("*.npy"))
    
    if not feature_files:
        print(f"Error: No feature .npy files found in {FEATURES_DIR}")
        print("Please run src/feature_extraction.py first.")
        return

    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

    for feature_path in tqdm(feature_files, desc="Compressing F0"):
        try:
            # 1. Create the new path
            relative_path = feature_path.relative_to(FEATURES_DIR)
            output_path = (COMPRESSED_DIR / relative_path)
            
            if output_path.exists():
                continue
            
            # 2. Load the original F0
            features = np.load(feature_path, allow_pickle=True).item()
            f0_raw = features['f0']
            
            # 3. Compress (using the numpy utility)
            compressed_data = compress_f0(f0_raw)
            
            # 4. Save the compressed dictionary
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, compressed_data, allow_pickle=True)
            
        except Exception as e:
            print(f"Error processing {feature_path.name}: {e}")
    
    print("\nF0 compression complete. ✨")


# --- Example Usage & Test ---
if __name__ == "__main__":
    
    # --- Test 1: Run the main compression task (Offline) ---
    main()

    # --- Test 2: Demonstrate the process (Offline & Online) ---
    print("\n--- Pitch Coder Differentiable Test (Simulated) ---")
    
    # 1. Create a dummy F0 signal (200 frames)
    n_frames = 200
    f0_np = np.zeros(n_frames, dtype=np.float32)
    f0_np[20:180] = 150 + np.random.rand(160) * 20 # Voiced segment
    
    f0_torch = torch.from_numpy(f0_np).float().unsqueeze(0) # (1, 200)
    
    # 2. Test the PyTorch module
    pitch_coder = PitchResidualCoder()
    f0_recon_torch = pitch_coder(f0_torch)
    f0_recon_np = f0_recon_torch.squeeze(0).numpy()
    
    # 3. Analyze results (using the offline numpy compressor for comparison)
    compressed_data_offline = compress_f0(f0_np)
    f0_reconstructed_offline = reconstruct_f0(compressed_data_offline)
    
    # 4. Error analysis
    voiced_mask = (f0_np != 0)
    
    # Error for the Differentiable Proxy
    rmse_proxy = np.sqrt(np.mean((f0_np[voiced_mask] - f0_recon_np[voiced_mask])**2))
    
    print(f"Original F0 shape:   {f0_torch.shape}")
    print(f"Reconstructed F0 shape: {f0_recon_torch.shape}")
    print(f"RMSE (Offline DCT Coder): {np.sqrt(np.mean((f0_np[voiced_mask] - f0_reconstructed_offline[voiced_mask])**2)):.4f} Hz")
    print(f"RMSE (Differentiable Proxy): {rmse_proxy:.4f} Hz (This drives training)")
    print("\n✅ PitchResidualCoder defined and ready for src/train_model.py.")