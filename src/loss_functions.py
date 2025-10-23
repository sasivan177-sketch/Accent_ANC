import torch
import torch.nn as nn
import torchcrepe
import torch.nn.functional as F

# --- Configuration (Adjust as needed) ---
SAMPLE_RATE = 16000 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------------------------------------------

class PitchLoss(nn.Module):
    """
    Computes pitch-related loss components using the CREPE model.
    Fixes: TypeError by ensuring the model is loaded once and passed correctly.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, device=DEVICE):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        
        # FIX: Load the CREPE model ONCE during initialization.
        self.pitch_model = torchcrepe.get_pretrained_model(model_capacity='full').to(self.device)
        self.pitch_model.eval() # Set to evaluation mode; we don't train CREPE

    def forward(self, y_hat: torch.Tensor, f0_target: torch.Tensor):
        """
        Calculates predicted pitch features.
        
        Args:
            y_hat: Predicted audio waveform tensor (B, T) or (B, 1, T).
            f0_target: Target fundamental frequency tensor.
        """
        
        # Prepare input: CREPE expects (B, T)
        y_hat_flat = y_hat.squeeze(1) if y_hat.dim() == 3 else y_hat

        # CORRECTED CALL to torchcrepe.predict
        with torch.no_grad(): # Don't track gradients for the feature extractor
            # Pass the loaded model object using the 'model' keyword
            f0_hat_salience = torchcrepe.predict(
                y_hat_flat, 
                self.sample_rate, 
                model=self.pitch_model, # <-- THE FIX IS HERE
                device=self.device,
                batch_size=512, 
                viterbi=True # Use Viterbi for smoother pitch
            )
        
        f0_hat = f0_hat_salience[0]
        salience = f0_hat_salience[1]
        
        # Resize f0_target to match f0_hat's time dimension for loss calculation
        # Uses interpolation, which is common but implementation dependent
        f0_target_resized = F.interpolate(
            f0_target.unsqueeze(1), size=f0_hat.size(-1), mode='linear', align_corners=False
        ).squeeze(1)

        # Return predicted F0, salience (confidence), and resized target F0
        return f0_hat, salience, f0_target_resized


class GeneratorLossFn(nn.Module):
    """
    Wrapper for all Generator loss components (Adversarial, Feature Matching, Pitch, etc.)
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        # Initialize the fixed Pitch Loss module
        self.pitch_loss_fn = PitchLoss(device=device) 
        
        # Placeholder for other loss components (you must define these)
        self.l1_loss = nn.L1Loss()
        
    def generator_loss(self, y_hat, f0_original, y, d_outputs_r, d_outputs_g):
        """
        Calculates the total Generator loss.
        y_hat: Predicted audio, f0_original: Target F0, y: Target audio
        d_outputs_r/g: Discriminator outputs for real/generated audio
        """
        
        # --- 1. Pitch Loss (Where the error occurred) ---
        f0_hat, salience, f0_target_resized = self.pitch_loss_fn(y_hat, f0_original.detach())
        
        # Weighted Mean Squared Error for Pitch
        loss_pitch = torch.sum(salience * (f0_hat - f0_target_resized)**2) / (torch.sum(salience) + 1e-6)
        
        # --- 2. Reconstruction Loss (L1/MSE on audio) ---
        loss_recon = self.l1_loss(y_hat, y) 

        # --- 3. Placeholder for Adversarial/Feature Matching Loss ---
        # Assuming d_outputs_g are the scores/features from the Discriminator for the generated audio
        loss_adv = -torch.mean(d_outputs_g[-1][-1]) # Example of non-saturating GAN loss
        loss_fm = torch.tensor(0.0) # Placeholder for Feature Matching Loss

        # --- 4. Total Loss ---
        loss_g = 1.0 * loss_recon + 1.0 * loss_pitch + 1.0 * loss_adv + 1.0 * loss_fm
        
        loss_dict = {
            "loss_g": loss_g.item(),
            "loss_recon": loss_recon.item(),
            "loss_pitch": loss_pitch.item(),
            "loss_adv": loss_adv.item(),
            "loss_fm": loss_fm.item()
        }
        
        return loss_g, loss_dict