import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_audio(file_path, sr=None):
    """
    Load an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
        sr (int, optional): Target sampling rate. Defaults to None (original sr)
    
    Returns:
        tuple: (audio array, sampling rate)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None, None

def plot_waveform(y, sr, title="Waveform"):
    """
    Plot the waveform of an audio signal.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title="Spectrogram"):
    """
    Plot the spectrogram of an audio signal.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        title (str): Title for the plot
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_audio_features(y, sr):
    """
    Extract basic audio features.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
    
    Returns:
        dict: Dictionary containing basic audio features
    """
    features = {}
    
    # Duration
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    
    # RMS energy
    features['rms'] = librosa.feature.rms(y=y)[0].mean()
    
    # Zero crossing rate
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    
    # Spectral centroid
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    
    # Spectral bandwidth
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    
    return features

def explore_audio_file(file_path):
    """
    Perform a complete exploration of an audio file.
    
    Args:
        file_path (str): Path to the audio file
    """
    print(f"\nExploring audio file: {file_path}")
    
    # Load audio
    y, sr = load_audio(file_path)
    if y is None:
        return
    
    # Print basic information
    print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    print(f"Number of samples: {len(y)}")
    
    # Plot waveform
    plot_waveform(y, sr, f"Waveform - {os.path.basename(file_path)}")
    
    # Plot spectrogram
    plot_spectrogram(y, sr, f"Spectrogram - {os.path.basename(file_path)}")
    
    # Get and print features
    features = get_audio_features(y, sr)
    print("\nAudio Features:")
    for feature, value in features.items():
        print(f"{feature}: {value:.4f}")

if __name__ == "__main__":
    # Define path to audio data directory
    data_dir = Path("data/audio")
    
    if not data_dir.exists():
        print(f"Error: Audio data directory not found at {data_dir}")
        print("Please ensure the directory exists and contains audio files")
    else:
        # Get list of audio files in the directory
        audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))
        
        if not audio_files:
            print(f"No audio files found in {data_dir}")
            print("Please place your .wav or .mp3 files in this directory")
        else:
            print(f"Found {len(audio_files)} audio files in {data_dir}")
            
            # Process each audio file
            for audio_file in audio_files:
                print(f"\nProcessing: {audio_file.name}")
                explore_audio_file(str(audio_file))
