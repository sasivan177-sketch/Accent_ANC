import pandas as pd
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# --- Configuration ---
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
TARGET_SR = 16000
TARGET_LOUDNESS = -23.0  # Target loudness in LUFS (Loudness Units Full Scale)

# Create a loudness meter
METER = pyln.Meter(TARGET_SR)


def process_audio(input_path: Path, output_path: Path):
    """
    Loads an audio file, applies all preprocessing steps, and saves the result.
    """
    if output_path.exists():
        # Skip if this file has already been processed
        return

    try:
        # 1. Load, convert to mono, and resample to 16 kHz
        # Using warnings.catch_warnings to suppress Librosa's Audioread warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        # 2. Apply silence trimming
        # top_db=30 means any signal 30dB below the max is considered silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        if len(y_trimmed) == 0:
            # Handle case where trimming removed the entire file
            # print(f"Warning: Silence trimming removed all audio from {input_path.name}")
            return

        # 3. Apply loudness normalization
        try:
            # Measure the loudness of the trimmed audio
            loudness = METER.integrated_loudness(y_trimmed)
            
            # Normalize to the target loudness
            y_normalized = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LOUDNESS)
            
            # Check for clipping and peak-normalize if necessary
            if np.max(np.abs(y_normalized)) > 1.0:
                y_normalized = y_normalized / np.max(np.abs(y_normalized))
                
        except Exception:
            # Fallback for very short files that pyloudnorm can't handle
            y_normalized = y_trimmed / (np.max(np.abs(y_trimmed)) + 1e-6)


        # 4. Save the processed file
        # Ensure the output directory for the speaker exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y_normalized, TARGET_SR)

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")


def main():
    """
    Main function to find all audio files from the CSVs and process them.
    """
    print("Starting audio preprocessing...")
    print(f"  Raw data source: {RAW_DATA_DIR}")
    print(f"  Processed data destination: {PROCESSED_DATA_DIR}")
    print(f"  Target sample rate: {TARGET_SR} Hz")
    print(f"  Target loudness: {TARGET_LOUDNESS} LUFS")
    
    # We use the CSV files as the source of truth
    metadata_files = [
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "test_full.csv"
    ]

    for meta_file_path in metadata_files:
        # Ensure the variable is a Path object before calling .exists()
        meta_file = Path(meta_file_path) 

        if not meta_file.exists():
            print(f"Warning: Metadata file not found at {meta_file}. Skipping...")
            continue

        print(f"\nProcessing files from {meta_file.name}...")
        df = pd.read_csv(meta_file)
        
        # Create a list of tasks to process
        tasks = []
        for _, row in df.iterrows():
            # e.g., "train/aew/arctic_a0001.wav"
            relative_path = Path(row['file_path'])
            speaker = row['speaker']
            
            # e.g., data/raw/train/aew/arctic_a0001.wav
            input_file_path = RAW_DATA_DIR / relative_path
            
            # e.g., data/processed/aew/arctic_a0001.wav
            output_file_path = PROCESSED_DATA_DIR / speaker / relative_path.name
            
            if not input_file_path.exists():
                # print(f"Warning: Source file not found: {input_file_path}")
                continue
                
            tasks.append((input_file_path, output_file_path))

        # Process all tasks with a progress bar
        for input_path, output_path in tqdm(tasks, desc=f"Processing {meta_file.stem}"):
            process_audio(input_path, output_path)

    print("\nAudio preprocessing complete. ✨")


if __name__ == "__main__":
    # Add these libraries to your requirements.txt if you haven't yet
    # pip install pandas librosa soundfile pyloudnorm tqdm
    main()