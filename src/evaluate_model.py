import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
import json
from typing import Dict, Any, Tuple

# --- 1. Import Project Modules ---
# Note: PESQ/STOI/MOSNet require external libraries; we use placeholders for simplicity.
from .encoder import Encoder
from .rvq_quantizer import RVQQuantizer, BITS_PER_QUANTIZER, FRAMES_PER_SEC
from .accent_encoder import AccentEncoder, ACCENT_DOWNSAMPLE_STRIDE, ACCENT_VQ_QUANTIZERS, ACCENT_VQ_CODEBOOK_SIZE
from .pitch_residual_coding import PitchResidualCoder
from .decoder_vocoder import DecoderVocoder
from .loss_functions import MelSpectrogram # Use MelSpectrogram utility

# --- Configuration ---
TARGET_SR = 16000
HOP_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory Setup ---
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR.mkdir(exist_ok=True)


# --- 2. Metric Calculation Functions ---

def calculate_bitrate_kbps(T_mel: int, T_wav: int, rvq_kbps: int) -> float:
    """Calculates the total effective bitrate of the codec."""
    
    # 1. RVQ Bitrate (Spectral Latent Codes)
    # This is determined by the configuration parameter (e.g., 8 kbps)
    B_rvq = rvq_kbps * 1000 # bits per second
    
    # 2. Accent Bitrate (Side Channel Codes)
    accent_fps = FRAMES_PER_SEC / ACCENT_DOWNSAMPLE_STRIDE
    bits_per_accent_frame = ACCENT_VQ_QUANTIZERS * int(np.log2(ACCENT_VQ_CODEBOOK_SIZE)) # e.g., 8*8 = 64 bits
    B_accent = bits_per_accent_frame * accent_fps # bits per second (e.g., ~333 bps)
    
    # 3. Pitch Bitrate (DCT Coefficients)
    # This is hard to quantify without storing the exact number of non-zero coeffs,
    # so we assume a small, constant overhead based on prior analysis (e.g., 0.5 kbps)
    B_pitch_overhead = 500 # 500 bits per second (0.5 kbps)
    
    B_total = B_rvq + B_accent + B_pitch_overhead
    
    return B_total / 1000.0 # Return in kbps


def calculate_metrics(y_real: torch.Tensor, y_hat: torch.Tensor, 
                      mel_extractor: nn.Module, spk_embedder: nn.Module, 
                      rvq_kbps: int) -> Dict[str, float]:
    """Calculates objective metrics for one pair of real and reconstructed waveforms."""
    
    # Ensure inputs are (1, T_wav)
    y_real = y_real.squeeze(0)
    y_hat = y_hat.squeeze(0)

    # --- 1. Spectral Distortion (L1 Mel Loss) ---
    mel_real = mel_extractor(y_real.unsqueeze(0))
    mel_fake = mel_extractor(y_hat.unsqueeze(0))
    # Detach y_real's mel for objective metric calculation
    mel_distortion = F.l1_loss(mel_fake, mel_real.detach()).item()

    # --- 2. Accent/Speaker Similarity (Cosine Similarity) ---
    # Extract speaker embeddings from both real and fake mel-spectrograms
    emb_real = spk_embedder(mel_real.detach())
    emb_fake = spk_embedder(mel_fake.detach())
    
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # Norms are calculated in the dot product if normalized beforehand.
    similarity = F.cosine_similarity(emb_real, emb_fake).item()

    # --- 3. Objective Quality Placeholders ---
    # NOTE: Actual calculation requires complex packages and is time-consuming.
    pesq_score = 1.0 + np.random.rand() * 1.5 # Placeholder: 1.0 to 2.5
    stoi_score = 0.5 + np.random.rand() * 0.2 # Placeholder: 0.5 to 0.7
    mos_score = 2.0 + np.random.rand() * 1.5 # Placeholder: 2.0 to 3.5

    # --- 4. Bitrate ---
    # Assuming T_mel and T_wav are approximately correct for a typical segment
    T_wav = y_real.shape[-1]
    T_mel = mel_real.shape[-1]
    total_kbps = calculate_bitrate_kbps(T_mel, T_wav, rvq_kbps)
    
    return {
        'mel_distortion_l1': mel_distortion,
        'accent_similarity_cos': similarity,
        'PESQ_score_ph': pesq_score,
        'STOI_score_ph': stoi_score,
        'MOS_score_ph': mos_score,
        'total_bitrate_kbps': total_kbps
    }


# --- 3. Main Evaluation Loop ---

def evaluate(model_path: Path, features_dir: Path, processed_dir: Path, rvq_kbps: int = 8):
    """
    Loads a trained model and evaluates performance on all available test data.
    """
    print(f"Starting evaluation of model: {model_path.name} at {rvq_kbps} kbps.")
    
    # --- 1. Load Model and Components ---
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Initialize components (must match dimensions used in training)
    Z_DIM = 128
    E_A_DIM = 128
    N_MELS = 80
    
    encoder = Encoder(in_channels=N_MELS, latent_dim=Z_DIM).to(DEVICE)
    rvq = RVQQuantizer(in_channels=Z_DIM).to(DEVICE)
    accent_encoder = AccentEncoder(in_channels=N_MELS).to(DEVICE)
    pitch_coder = PitchResidualCoder().to(DEVICE) 
    generator = DecoderVocoder(z_dim=Z_DIM, e_a_dim=E_A_DIM).to(DEVICE)
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    rvq.load_state_dict(checkpoint['rvq_state_dict'])
    accent_encoder.load_state_dict(checkpoint['accent_encoder_state_dict'])
    pitch_coder.load_state_dict(checkpoint['pitch_coder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    encoder.eval()
    rvq.eval()
    accent_encoder.eval()
    pitch_coder.eval()
    generator.eval()
    
    # Initialize metric calculators
    mel_extractor = MelSpectrogram().to(DEVICE).eval()
    spk_embedder = SpeakerAccentEmbedder().to(DEVICE).eval()
    
    # --- 2. Find Evaluation Files ---
    # We assume 'test' features are distinct from 'train'
    eval_files = [f for f in features_dir.rglob("*.npy") if "test" in f.parts or "test_full" in f.parts]
    if not eval_files:
        print("Warning: No 'test' feature files found. Evaluating on first 100 features.")
        eval_files = list(features_dir.rglob("*.npy"))[:100]

    results = []
    
    with torch.no_grad():
        for feature_path in tqdm(eval_files, desc=f"Evaluating ({len(eval_files)} files)"):
            try:
                # 3. Load Data
                features = np.load(feature_path, allow_pickle=True).item()
                f0_orig_np = features['f0']
                mel_orig_np = features['mel_spectrogram']
                
                # Find corresponding real WAV file
                relative_path = feature_path.relative_to(features_dir)
                wav_path = (processed_dir / relative_path).with_suffix(".wav")
                
                y_real_np, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                
                # Convert to Tensors (unsqueeze for batch=1)
                f0_orig = torch.from_numpy(f0_orig_np).float().unsqueeze(0).to(DEVICE) # (1, T_mel)
                mel_orig = torch.from_numpy(mel_orig_np).float().unsqueeze(0).to(DEVICE) # (1, N_MELS, T_mel)
                y_real = torch.from_numpy(y_real_np).float().unsqueeze(0).to(DEVICE) # (1, T_wav)

                # 4. Codec Forward Pass (Compression & Reconstruction)
                z = encoder(mel_orig)
                z_q, _ = rvq(z, bitrate_kbps=rvq_kbps) # Use target bitrate
                e_a, _ = accent_encoder(mel_orig)
                f0_recon = pitch_coder(f0_orig)
                
                # Reconstruct waveform
                y_hat = generator(z_q, e_a, f0_recon.unsqueeze(1).squeeze(1)) # (1, 1, T_wav)

                # Trim y_real to match y_hat length
                y_real = y_real[:, :y_hat.size(-1)]
                
                # 5. Calculate Metrics
                metrics = calculate_metrics(y_real.cpu(), y_hat.cpu(), 
                                            mel_extractor.cpu(), spk_embedder.cpu(), 
                                            rvq_kbps)
                
                # 6. Save Reconstructed Audio (Optional, for listening tests)
                # Ensure the output directory structure is mirrored
                output_wav_path = (Path("data/reconstructed") / relative_path).with_suffix(".wav")
                output_wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_wav_path, y_hat.squeeze().cpu().numpy(), TARGET_SR)

                metrics['file_path'] = str(relative_path)
                metrics['bitrate_setting'] = rvq_kbps
                results.append(metrics)
                
            except Exception as e:
                print(f"Error processing file {feature_path.name}: {e}")

    # --- 7. Save Results ---
    df = pd.DataFrame(results)
    
    # Calculate Averages and Save
    avg_metrics = df.drop(columns=['file_path', 'bitrate_setting']).mean().to_dict()
    avg_metrics['model'] = model_path.name
    avg_metrics['bitrate_setting'] = rvq_kbps
    
    metrics_file = REPORTS_DIR / "metrics.csv"
    if metrics_file.exists():
        # Append to existing file
        existing_df = pd.read_csv(metrics_file)
        final_df = pd.concat([existing_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    else:
        final_df = pd.DataFrame([avg_metrics])

    final_df.to_csv(metrics_file, index=False)
    
    print("\nEvaluation complete. Average metrics saved to reports/metrics.csv:")
    print(final_df.iloc[-1])
    
    # Save the detailed per-file results for analysis
    df.to_csv(REPORTS_DIR / f"detailed_metrics_{rvq_kbps}kbps.csv", index=False)


if __name__ == "__main__":
    # Example usage: Find the latest saved model and evaluate it at 8kbps
    # You would typically loop through bitrates like {2, 4, 8, 16}
    try:
        latest_model = max(MODELS_DIR.glob("joint_model_ep*.pth"), key=Path.getmtime)
        evaluate(latest_model, FEATURES_DIR, PROCESSED_DIR, rvq_kbps=8)
        
    except ValueError:
        print("Error: No model checkpoints found in 'models/' directory.")
        print("Please run src/train_model.py first.")