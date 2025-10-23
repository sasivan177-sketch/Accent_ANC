import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import List, Tuple, Dict, Any

# --- 1. Import Project Modules ---
from .encoder import Encoder
from .rvq_quantizer import RVQQuantizer
from .accent_encoder import AccentEncoder
from .pitch_residual_coding import PitchResidualCoder 
from .decoder_vocoder import DecoderVocoder
from .loss_functions import CompositeLoss

# --- 2. Placeholder Discriminators (FIXED: Registering parameters) ---
class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        print("Note: Using placeholder MultiPeriodDiscriminator.")
        # FIX: Explicitly define a layer that has trainable parameters
        self.conv = nn.Conv1d(1, 1, 1) 
    def forward(self, x):
        # FIX: Output feature list directly (not wrapped in another list)
        score = torch.randn(x.size(0), 1, requires_grad=True).to(x.device)
        features = [torch.randn(x.size(0), 1, requires_grad=True).to(x.device)]
        return [score], features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        print("Note: Using placeholder MultiScaleDiscriminator.")
        # FIX: Explicitly define a layer that has trainable parameters
        self.conv = nn.Conv1d(1, 1, 1)
    def forward(self, x):
        # FIX: Output feature list directly (not wrapped in another list)
        score = torch.randn(x.size(0), 1, requires_grad=True).to(x.device)
        features = [torch.randn(x.size(0), 1, requires_grad=True).to(x.device)]
        return [score], features

# --- 3. Custom Dataset Loader (FIXED TENSOR DIMS) ---
class FeatureDataset(Dataset):
    def __init__(self, features_dir: Path, processed_dir: Path, segment_len: int = 16000):
        self.features_dir = Path(features_dir)
        self.processed_dir = Path(processed_dir)
        self.segment_len = segment_len
        self.data_list = self._create_data_list()
        if not self.data_list:
             warnings.warn(f"Dataset is empty! Checked {self.features_dir} for .npy files.", UserWarning)

    def _create_data_list(self):
        data = []
        for feature_path in self.features_dir.rglob("*.npy"):
            relative_path = feature_path.relative_to(self.features_dir)
            wav_path = (self.processed_dir / relative_path).with_suffix(".wav")
            if wav_path.exists():
                data.append((feature_path, wav_path))
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        feature_path, wav_path = self.data_list[idx]
        features = np.load(feature_path, allow_pickle=True).item()
        f0_original = torch.from_numpy(features['f0']).float()
        mel_spectrogram = torch.from_numpy(features['mel_spectrogram']).float()
        
        y_np, sr = librosa.load(wav_path, sr=None, mono=True)
        wav = torch.from_numpy(y_np).float().unsqueeze(0)
        
        T_wav = wav.shape[-1]
        
        if T_wav < self.segment_len:
            padding_needed = self.segment_len - T_wav
            wav = F.pad(wav, (0, padding_needed))
            T_wav = self.segment_len
        
        start_wav = np.random.randint(0, T_wav - self.segment_len + 1)
        wav_segment = wav[:, start_wav : start_wav + self.segment_len]

        hop_length = 256 
        segment_mel_len = self.segment_len // hop_length
        start_mel = start_wav // hop_length
        T_mel = mel_spectrogram.shape[-1]
        end_mel = min(start_mel + segment_mel_len, T_mel)
        mel_segment = mel_spectrogram[:, start_mel:end_mel]
        f0_segment = f0_original[start_mel:end_mel]

        if mel_segment.shape[-1] < segment_mel_len:
            padding_needed = segment_mel_len - mel_segment.shape[-1]
            mel_segment = F.pad(mel_segment, (0, padding_needed))
            f0_segment = F.pad(f0_segment, (0, padding_needed))

        # FIX: return the 2D segment (1, T_wav_seg). 
        return mel_segment, f0_segment, wav_segment 


# --- 4. Training Configuration (Targeting 30 Minutes) ---
BATCH_SIZE = 32 
LEARNING_RATE = 2e-4
EPOCHS = 1
SAVE_INTERVAL = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_DIM = 128
E_A_DIM = 128
N_MELS = 80
SEGMENT_LEN_WAV = 16000


def train(features_dir: Path, processed_dir: Path, models_dir: Path):
    print(f"Starting training on device: {DEVICE} with BATCH_SIZE={BATCH_SIZE} and EPOCHS={EPOCHS}")

    # --- Data Setup ---
    train_dataset = FeatureDataset(features_dir, processed_dir, segment_len=SEGMENT_LEN_WAV)
    if len(train_dataset) == 0:
        print("Error: Dataset is empty. Check paths and feature files.")
        return
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    # --- Model Initialization ---
    encoder = Encoder(in_channels=N_MELS, latent_dim=Z_DIM).to(DEVICE)
    rvq = RVQQuantizer(in_channels=Z_DIM).to(DEVICE)
    accent_encoder = AccentEncoder(in_channels=N_MELS).to(DEVICE)
    pitch_coder = PitchResidualCoder().to(DEVICE) 
    generator = DecoderVocoder(z_dim=Z_DIM, e_a_dim=E_A_DIM).to(DEVICE)
    
    mpd = MultiPeriodDiscriminator().to(DEVICE)
    msd = MultiScaleDiscriminator().to(DEVICE)
    # The optimizers are happy now because mpd/msd have registered parameters
    discriminator = nn.ModuleList([mpd, msd]).to(DEVICE) 

    loss_fn = CompositeLoss(accent_encoder=accent_encoder, device=DEVICE)
    
    optim_G = optim.AdamW(
        list(encoder.parameters()) + list(rvq.parameters()) + 
        list(accent_encoder.parameters()) + list(pitch_coder.parameters()) + 
        list(generator.parameters()),
        lr=LEARNING_RATE, betas=(0.8, 0.99)
    )
    optim_D = optim.AdamW(
        discriminator.parameters(),
        lr=LEARNING_RATE, betas=(0.8, 0.99)
    )

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        G_loss_total = 0
        D_loss_total = 0
        
        tqdm_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for mel, f0_orig, y in tqdm_loop:
            mel, f0_orig, y = mel.to(DEVICE), f0_orig.to(DEVICE), y.to(DEVICE)
            
            # --- A. Generator Step ---
            optim_G.zero_grad()
            
            z = encoder(mel)                      
            z_q, _ = rvq(z)                       
            e_a, _ = accent_encoder(mel)          
            f0_recon = pitch_coder(f0_orig)       
            
            y_hat = generator(z_q, e_a, f0_recon.unsqueeze(1).squeeze(1))
            
            disc_real_outputs = [d(y) for d in discriminator]
            disc_fake_outputs = [d(y_hat) for d in discriminator]
            
            disc_real_scores = [out[0] for out in disc_real_outputs]
            disc_real_feats = [out[1] for out in disc_real_outputs]
            disc_fake_scores = [out[0] for out in disc_fake_outputs] 
            disc_fake_feats = [out[1] for out in disc_fake_outputs] 
            
            loss_g, loss_dict = loss_fn.generator_loss(
                y=y, y_hat=y_hat, 
                f0_original=f0_orig, f0_recon=f0_recon, 
                disc_real_outputs=disc_real_scores, disc_fake_outputs=disc_fake_scores,
                disc_real_feats=disc_real_feats, disc_fake_feats=disc_fake_feats
            )
            
            loss_g.backward()
            optim_G.step()
            G_loss_total += loss_g.item()

            # --- B. Discriminator Step ---
            optim_D.zero_grad()
            
            disc_fake_outputs_d = [d(y_hat.detach()) for d in discriminator]
            disc_real_outputs_d = [d(y.detach()) for d in discriminator]
            
            disc_real_scores_d = [out[0] for out in disc_real_outputs_d]
            disc_fake_scores_d = [out[0] for out in disc_fake_outputs_d]
            
            loss_d = loss_fn.discriminator_loss(
                disc_real_scores_d, 
                disc_fake_scores_d
            )
            
            loss_d.backward()
            optim_D.step()
            D_loss_total += loss_d.item()
            
            tqdm_loop.set_postfix(
                G=f"{loss_g.item():.4f}", 
                D=f"{loss_d.item():.4f}",
                Mel=f"{loss_dict.get('L_Mel', 0):.4f}",
                Pitch=f"{loss_dict.get('L_Pitch', 0):.4f}",
                Spk=f"{loss_dict.get('L_Speaker', 0):.4f}"
            )

        # --- Logging and Checkpoint ---
        avg_g_loss = G_loss_total / len(train_loader)
        avg_d_loss = D_loss_total / len(train_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f}")
        
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = models_dir / f"joint_model_ep{epoch}.pth"
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'rvq_state_dict': rvq.state_dict(),
                'accent_encoder_state_dict': accent_encoder.state_dict(),
                'pitch_coder_state_dict': pitch_coder.state_dict(), 
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optim_G_state_dict': optim_G.state_dict(),
                'optim_D_state_dict': optim_D.state_dict(),
                'avg_g_loss': avg_g_loss,
            }, checkpoint_path)


if __name__ == "__main__":
    FEATURES_DIR = Path("data/features")
    PROCESSED_DIR = Path("data/processed")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    train(FEATURES_DIR, PROCESSED_DIR, MODELS_DIR)