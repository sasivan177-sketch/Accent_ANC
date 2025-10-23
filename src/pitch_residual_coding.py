import numpy as np
import scipy.fft
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

def compress_f0(f0: np.ndarray) -> dict:
    """
    Compresses an F0 contour using DCT on the residual.
    
    Args:
        f0 (np.ndarray): The raw F0 contour, shape (n_frames,).
        
    Returns:
        dict: A dictionary containing the compressed data:
              - 'sparse_dct_coeffs' (np.ndarray): The thresholded DCT coefficients.
              - 'mean_f0' (float): The mean of the voiced F0.
              - 'unvoiced_flags' (np.ndarray): Boolean array (True where f0==0).
    """
    unvoiced_flags = (f0 == 0)
    voiced_f0 = f0[~unvoiced_flags]

    # Handle files that are entirely unvoiced
    if len(voiced_f0) == 0:
        mean_f0 = 0.0
        f0_residual = f0
    else:
        # 1. Calculate the mean and the residual
        mean_f0 = np.mean(voiced_f0)
        # We create a new array for the residual
        f0_residual = f0.copy()
        # Only subtract the mean from the voiced parts
        f0_residual[~unvoiced_flags] -= mean_f0

    # 2. Apply Discrete Cosine Transform (DCT)
    # The residual signal (including zeros) is transformed.
    f0_dct = scipy.fft.dct(f0_residual, type=2, norm='ortho')

    # 3. Apply threshold to get sparse coefficients
    f0_dct_sparse = f0_dct.copy()
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
    Reconstructs the F0 contour from the compressed data.
    (This function will be used by the decoder later)
    """
    sparse_coeffs = compressed_data['sparse_dct_coeffs']
    mean_f0 = compressed_data['mean_f0']
    unvoiced_flags = compressed_data['unvoiced_flags']
    
    # 1. Apply Inverse DCT (IDCT)
    f0_residual_recon = scipy.fft.idct(sparse_coeffs, type=2, norm='ortho')
    
    # 2. Add the mean back to the voiced parts
    f0_recon = f0_residual_recon
    f0_recon[~unvoiced_flags] += mean_f0
    
    # 3. Re-apply unvoiced flags (zeros)
    # This also corrects any small non-zero values in unvoiced parts
    # that may have appeared from the IDCT.
    f0_recon[unvoiced_flags] = 0.0
    
    # Ensure no negative F0 values
    f0_recon[f0_recon < 0] = 0.0
    
    return f0_recon

def main():
    """
    Finds all feature .npy files, compresses their F0,
    and saves the compressed data.
    """
    print("Starting F0 (pitch) compression...")
    print(f"  Features source: {FEATURES_DIR}")
    print(f"  Compressed data destination: {COMPRESSED_DIR}")
    print(f"  DCT Threshold: {DCT_THRESHOLD}")

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
            
            # 3. Compress
            compressed_data = compress_f0(f0_raw)
            
            # 4. Save the compressed dictionary
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, compressed_data, allow_pickle=True)
            
        except Exception as e:
            print(f"Error processing {feature_path.name}: {e}")
    
    print("\nF0 compression complete. ✨")


# --- Example Usage & Test ---
if __name__ == "__main__":
    
    # --- Test 1: Run the main compression task ---
    main()

    # --- Test 2: Demonstrate the process ---
    print("\n--- Compression Test ---")
    # 1. Create a dummy F0 signal (200 frames)
    # A 150Hz sine wave with some noise, plus an unvoiced segment
    n_frames = 200
    t = np.linspace(0, 8 * np.pi, n_frames)
    f0_original = 150 + np.sin(t) * 20 + np.random.randn(n_frames) * 0.5
    f0_original[80:110] = 0.0 # Add an unvoiced segment
    
    # 2. Compress
    compressed_data = compress_f0(f0_original)
    
    # 3. Reconstruct
    f0_reconstructed = reconstruct_f0(compressed_data)
    
    # 4. Analyze results
    original_coeffs = scipy.fft.dct(f0_original, type=2, norm='ortho')
    sparse_coeffs = compressed_data['sparse_dct_coeffs']
    
    original_nzc = np.sum(np.abs(original_coeffs) > 0.0)
    compressed_nzc = np.sum(np.abs(sparse_coeffs) > 0.0)
    
    # Calculate error only on voiced parts
    voiced_mask = (f0_original != 0)
    rmse = np.sqrt(np.mean((f0_original[voiced_mask] - f0_reconstructed[voiced_mask])**2))
    
    print(f"Original F0 (frames):   {len(f0_original)}")
    print(f"Non-zero DCT coeffs (Original): {original_nzc}")
    print(f"Non-zero DCT coeffs (Sparse):   {compressed_nzc}")
    print(f"Compression (coeffs): {original_nzc / compressed_nzc:.2f}x")
    print(f"Reconstruction RMSE (voiced): {rmse:.4f} Hz")