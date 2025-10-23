import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe
import os
import sys

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 16000
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
LOG_INTERVAL = 100

# --- 1. LOSS FUNCTIONS (Copied and fixed from previous step) ---

# The PitchLoss class contains the fix for the TypeError: predict()
class PitchLoss(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, device=DEVICE):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        # FIX: Load the CREPE model ONCE
        self.pitch_model = torchcrepe.get_pretrained_model(model_capacity='full').to(self.device)
        self.pitch_model.eval() 

    def forward(self, y_hat: torch.Tensor, f0_target: torch.Tensor):
        y_hat_flat = y_hat.squeeze(1) if y_hat.dim() == 3 else y_hat

        with torch.no_grad():
            # CORRECTED CALL: Pass the loaded model object
            f0_hat_salience = torchcrepe.predict(
                y_hat_flat, 
                self.sample_rate, 
                model=self.pitch_model, 
                device=self.device,
                batch_size=512, 
                viterbi=True
            )
        
        f0_hat = f0_hat_salience[0]
        salience = f0_hat_salience[1]
        
        # Resize f0_target to match f0_hat's time dimension
        f0_target_resized = F.interpolate(
            f0_target.unsqueeze(1), size=f0_hat.size(-1), mode='linear', align_corners=False
        ).squeeze(1)

        return f0_hat, salience, f0_target_resized

class GeneratorLossFn(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.pitch_loss_fn = PitchLoss(device=device) 
        self.l1_loss = nn.L1Loss()
        
    def generator_loss(self, y_hat, f0_original, y, d_outputs_g, d_outputs_r):
        """Calculates total Generator loss (Adversarial, Reconstruction, Pitch)."""
        
        # --- Pitch Loss ---
        f0_hat, salience, f0_target_resized = self.pitch_loss_fn(y_hat, f0_original.detach())
        loss_pitch = torch.sum(salience * (f0_hat - f0_target_resized)**2) / (torch.sum(salience) + 1e-6)
        
        # --- Reconstruction Loss (L1) ---
        loss_recon = self.l1_loss(y_hat, y) 

        # --- Adversarial Loss (Non-saturating GAN loss) ---
        # d_outputs_g is typically a list of lists: [[score_1, fm_1], [score_2, fm_2], ...]
        adv_scores_g = [d_output[0] for d_output in d_outputs_g]
        loss_adv = sum([-torch.mean(score) for score in adv_scores_g])

        # --- Feature Matching Loss (L1 on intermediate features) ---
        fm_losses = []
        for d_output_g, d_output_r in zip(d_outputs_g, d_outputs_r):
             for fm_g, fm_r in zip(d_output_g[1], d_output_r[1]):
                 # Detach real features to use them as fixed targets
                 fm_losses.append(self.l1_loss(fm_g, fm_r.detach()))
        loss_fm = sum(fm_losses) / len(fm_losses) if fm_losses else torch.tensor(0.0).to(DEVICE)


        # --- Total Generator Loss ---
        # Weights are crucial hyperparameters here
        loss_g = 1.0 * loss_recon + 1.0 * loss_pitch + 1.0 * loss_adv + 2.0 * loss_fm
        
        loss_dict = {
            "G/total": loss_g.item(),
            "G/recon": loss_recon.item(),
            "G/pitch": loss_pitch.item(),
            "G/adv": loss_adv.item(),
            "G/fm": loss_fm.item()
        }
        
        return loss_g, loss_dict

# --- 2. MODEL PLACEHOLDERS ---
# Note: In a real project, these would be imported from src/discriminator.py
class DiscriminatorLossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def discriminator_loss(self, d_outputs_r, d_outputs_g):
        """Calculates total Discriminator loss."""
        
        loss_r = 0.0
        loss_g = 0.0
        
        # Real scores should be close to 1
        for d_output in d_outputs_r:
            score_r = d_output[0]
            loss_r += self.mse_loss(score_r, torch.ones_like(score_r))
        
        # Generated scores should be close to 0
        for d_output in d_outputs_g:
            score_g = d_output[0]
            loss_g += self.mse_loss(score_g, torch.zeros_like(score_g))
            
        loss_d = loss_r + loss_g
        
        loss_dict = {
            "D/total": loss_d.item(),
            "D/real": loss_r.item(),
            "D/gen": loss_g.item()
        }
        return loss_d, loss_dict

class Generator(nn.Module):
    # The model that takes features (x) and outputs audio (y_hat)
    def forward(self, x): return torch.randn(x.size(0), 1, 16000 * 4).to(x.device) # Mock: (B, 1, AudioLen)
class MultiPeriodDiscriminator(nn.Module):
    # The discriminator that judges generated audio (y_hat)
    def forward(self, y): 
        # Mock: return a list of (score, features) tuples for each sub-discriminator
        return [(torch.randn(y.size(0), 1).to(y.device), [torch.randn(1), torch.randn(1)])]
class MultiScaleDiscriminator(nn.Module):
    # Another discriminator
    def forward(self, y): 
        return [(torch.randn(y.size(0), 1).to(y.device), [torch.randn(1), torch.randn(1)])]


# --- 3. MOCK DATA LOADER ---
# Replaces actual data loading (Dataloader, Dataset)
def get_data_loader(batch_size, num_iterations):
    # Simulates a dataloader yielding (input_features, target_audio, target_f0)
    class MockDataLoader:
        def __iter__(self):
            for _ in range(num_iterations):
                # x: Input features (e.g., mel, accent embedding) - [B, Features, TimeSteps]
                x = torch.randn(batch_size, 80, 100).to(DEVICE)
                # y: Target audio - [B, 1, AudioSamples] (e.g., 4s @ 16kHz = 64000)
                y = torch.randn(batch_size, 1, SAMPLE_RATE * 4).to(DEVICE)
                # f0_original: Target F0 - [B, F0_TimeSteps]
                f0_original = torch.randn(batch_size, 100).to(DEVICE) 
                yield x, y, f0_original
    return MockDataLoader()

# --- 4. MAIN TRAINING FUNCTION ---

def train():
    # --- Initialization ---
    
    # Models
    generator = Generator().to(DEVICE)
    mpd = MultiPeriodDiscriminator().to(DEVICE) # MPD
    msd = MultiScaleDiscriminator().to(DEVICE) # MSD
    discriminators = [mpd, msd]
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    # Discriminator optimizer for all discriminators combined
    optimizer_d = torch.optim.Adam([*mpd.parameters(), *msd.parameters()], lr=LEARNING_RATE) 
    
    # Loss Functions
    loss_fn_g = GeneratorLossFn(device=DEVICE)
    loss_fn_d = DiscriminatorLossFn()
    
    # Data Loader
    num_iterations = 973 # Based on your log
    data_loader = get_data_loader(BATCH_SIZE, num_iterations)

    print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        
        generator.train()
        for d in discriminators: d.train()

        for i, (x, y, f0_original) in enumerate(data_loader, 1):
            
            # --- GENERATOR STEP ---
            optimizer_g.zero_grad()
            
            # 1. Forward Pass
            y_hat = generator(x)
            
            # 2. Discriminator outputs for generated (g) and real (r) audio
            d_outputs_g = [d(y_hat) for d in discriminators]
            # Need to detach y to prevent gradient flow during D loss calculation
            d_outputs_r = [d(y.detach()) for d in discriminators] 
            
            # 3. Calculate Generator Loss (uses corrected PitchLoss internally)
            loss_g, loss_dict_g = loss_fn_g.generator_loss(
                y_hat, f0_original.detach(), y, d_outputs_g, d_outputs_r
            )
            
            # 4. Backward Pass and Update
            loss_g.backward()
            optimizer_g.step()

            # --- DISCRIMINATOR STEP ---
            optimizer_d.zero_grad()
            
            # Re-run D on generated audio (must be done after G's update)
            # Use y_hat.detach() to prevent gradient flow to G
            d_outputs_g_detached = [d(y_hat.detach()) for d in discriminators]
            d_outputs_r_original = [d(y) for d in discriminators] # Re-run D on real audio

            # 1. Calculate Discriminator Loss
            loss_d, loss_dict_d = loss_fn_d.discriminator_loss(
                d_outputs_r_original, d_outputs_g_detached
            )
            
            # 2. Backward Pass and Update
            loss_d.backward()
            optimizer_d.step()
            
            # --- LOGGING ---
            if i % LOG_INTERVAL == 0 or i == num_iterations:
                print(
                    f"Epoch {epoch}/{EPOCHS} | Iter {i}/{num_iterations} | "
                    f"G Loss: {loss_dict_g['G/total']:.3f} (P: {loss_dict_g['G/pitch']:.3f}, R: {loss_dict_g['G/recon']:.3f}) | "
                    f"D Loss: {loss_dict_d['D/total']:.3f}"
                )

        # --- END OF EPOCH ---
        print(f"\n--- Epoch {epoch} Finished ---\n")
        # Save checkpoints here
        # torch.save(generator.state_dict(), f"generator_{epoch}.pt")


if __name__ == "__main__":
    # Ensure torchcrepe is downloaded
    # Note: Sometimes torchcrepe requires a pre-download step
    try:
        if not os.path.exists(os.path.expanduser('~/.cache/torch/crepe/full.pth')):
             print("Downloading torchcrepe model weights...")
             torchcrepe.get_pretrained_model(model_capacity='full')
    except Exception as e:
         print(f"Warning: Could not pre-download CREPE weights. It may download on first use. Error: {e}")
         
    # Call the main training function
    train()