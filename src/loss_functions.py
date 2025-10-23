import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import List, Tuple, Dict, Any

# --- Configuration Constants ---
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
WIN_LENGTH = 1024

# --- Helper Module: MelSpectrogram Extractor ---
class MelSpectrogram(nn.Module):
    """Computes Mel-spectrograms for waveform tensors using torch.stft and librosa filter bank."""
    def __init__(self, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                 n_mels=N_MELS, win_length=WIN_LENGTH):
        super().__init__()
        
        mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr/2
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = nn.Parameter(torch.hann_window(win_length), requires_grad=False)
        self.register_buffer('min_val', torch.tensor(1e-5))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Input: (B, 1, T_wav). Output: (B, N_MELS, T_mel)"""
        
        if y.dim() == 3 and y.size(1) == 1:
            y = y.squeeze(1) # Shape is now (B, T_wav)
        
        stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        magnitude = torch.abs(stft) ** 2

        mel_spec = torch.matmul(self.mel_basis, magnitude)
        mel = torch.log(torch.clamp(mel_spec, min=self.min_val))
        return mel


# --- Placeholder for External Speaker/Accent Embedding Extractor ---
class SpeakerAccentEmbedder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(N_MELS, embed_dim, kernel_size=1) 
        print("Note: Using placeholder SpeakerAccentEmbedder.")
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.proj(mel)
        embedding = x.mean(dim=-1) 
        return embedding


class CompositeLoss(nn.Module):
    def __init__(self, accent_encoder: nn.Module, device: torch.device):
        super().__init__()
        
        self.w_mel = 45.0
        self.w_adv_g = 1.0
        self.w_fm = 2.0
        self.w_pitch = 10.0
        self.w_spk_con = 5.0

        self.mel_extractor = MelSpectrogram().to(device) 
        self.spk_embedder = SpeakerAccentEmbedder(embed_dim=256).to(device)
        self.accent_encoder = accent_encoder 
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def _adversarial_loss_g(self, disc_fake_outputs: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Generator's desire to fool the discriminator (target=1).
        
        Args:
            disc_fake_outputs (List[List[torch.Tensor]]): A list where each element 
            is the [score_tensor] list from a discriminator (e.g., [[score_mpd], [score_msd]]).
        """
        loss = 0
        # disc_output_list is the inner list containing the score tensor, e.g., [score_tensor]
        for disc_output_list in disc_fake_outputs: 
            # ⭐️ FIX: Access the actual score tensor at index 0
            disc_score_tensor = disc_output_list[0] 
            loss += self.mse_loss(disc_score_tensor, torch.ones_like(disc_score_tensor))
        return loss
    
    def _feature_matching_loss(self, disc_real_feats: List[List[torch.Tensor]], 
                                disc_fake_feats: List[List[torch.Tensor]]) -> torch.Tensor:
        """Measures distance between discriminator feature maps for real/fake."""
        loss = 0
        # disc_feats is a list of lists: [MPD_feats, MSD_feats]
        for real_feats_list, fake_feats_list in zip(disc_real_feats, disc_fake_feats):
            for real_feat, fake_feat in zip(real_feats_list, fake_feats_list):
                loss += self.l1_loss(real_feat, fake_feat)
        return loss

    def generator_loss(self, 
                       y: torch.Tensor, y_hat: torch.Tensor, 
                       f0_original: torch.Tensor, f0_recon: torch.Tensor,
                       disc_real_outputs: List[List[torch.Tensor]], disc_fake_outputs: List[List[torch.Tensor]],
                       disc_real_feats: List[List[torch.Tensor]], disc_fake_feats: List[List[torch.Tensor]]
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        loss_dict = {}

        # 1. L1 Mel Loss 
        mel_real = self.mel_extractor(y.detach())
        mel_fake = self.mel_extractor(y_hat)
        L_mel = self.l1_loss(mel_fake, mel_real) * self.w_mel
        
        # 2. Adversarial Loss (Uses fixed function)
        L_adv_g = self._adversarial_loss_g(disc_fake_outputs) * self.w_adv_g
        
        # 3. Feature Matching Loss 
        L_fm = self._feature_matching_loss(disc_real_feats, disc_fake_feats) * self.w_fm

        # 4. Pitch Consistency Loss
        L_pitch = self.l1_loss(f0_recon, f0_original) * self.w_pitch
        
        # 5. Contrastive Accent Loss (Speaker Identity)
        mel_orig_spk_emb = self.spk_embedder(mel_real)
        mel_fake_spk_emb = self.spk_embedder(mel_fake)
        L_spk_con = self.mse_loss(mel_fake_spk_emb, mel_orig_spk_emb.detach()) * self.w_spk_con
        
        # 6. VQ Codebook Loss (Placeholder)
        L_vq_commit = torch.tensor(0.0, device=y.device) 

        # --- Total Generator Loss ---
        L_g_total = L_mel + L_adv_g + L_fm + L_pitch + L_spk_con + L_vq_commit

        loss_dict.update({
            'L_Mel': L_mel.item(),
            'L_Adv_G': L_adv_g.item(),
            'L_FM': L_fm.item(),
            'L_Pitch': L_pitch.item(),
            'L_Speaker': L_spk_con.item(),
            'L_VQ_Commit': L_vq_commit.item(),
        })

        return L_g_total, loss_dict

    def discriminator_loss(self, disc_real_outputs: List[List[torch.Tensor]], 
                           disc_fake_outputs: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Computes the total loss for the Discriminator.
        """
        L_d_total = 0
        
        # Iterate over lists of scores from D1, D2, ...
        for real_output_list, fake_output_list in zip(disc_real_outputs, disc_fake_outputs):
            # Access the score tensor at index 0
            real_score = real_output_list[0] 
            fake_score = fake_output_list[0]
            
            # Loss for Real (D wants to correctly classify real as 1)
            loss_real = self.mse_loss(real_score, torch.ones_like(real_score))
            # Loss for Fake (D wants to correctly classify fake as 0)
            loss_fake = self.mse_loss(fake_score, torch.zeros_like(fake_score))
            
            L_d_total += (loss_real + loss_fake)
            
        return L_d_total