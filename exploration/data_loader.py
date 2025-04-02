import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path

class AudioDataset(Dataset):
    """Custom Dataset for loading audio files."""
    
    def __init__(self, data_dir, sample_rate=22050, duration=None, transform=None):
        """
        Args:
            data_dir (str): Directory with all the audio files
            sample_rate (int): Target sample rate for audio files
            duration (float): Duration in seconds to load from each file (None for full file)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        
        # Get all audio files
        self.file_list = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
        
        if not self.file_list:
            raise RuntimeError(f"No audio files found in {data_dir}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load and return an audio sample."""
        audio_path = str(self.file_list[idx])
        
        # Load audio with specified duration if provided
        if self.duration:
            samples = int(self.duration * self.sample_rate)
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        else:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(y)
        
        # Apply transform if provided
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        
        return audio_tensor

def create_dataloader(data_dir, batch_size=32, sample_rate=22050, duration=None, 
                     transform=None, shuffle=True, num_workers=4):
    """
    Create a DataLoader for audio files.
    
    Args:
        data_dir (str): Directory containing audio files
        batch_size (int): How many samples per batch to load
        sample_rate (int): Target sample rate for audio files
        duration (float): Duration in seconds to load from each file (None for full file)
        transform (callable, optional): Optional transform to be applied on samples
        shuffle (bool): Whether to shuffle the data
        num_workers (int): How many subprocesses to use for data loading
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = AudioDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        duration=duration,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

if __name__ == "__main__":
    # Example usage
    data_dir = "data/audio"
    
    # Create a simple transform pipeline
    transform = torch.nn.Sequential(
        # Add any preprocessing steps here
        # Example: normalize the audio
        torch.nn.Lambda(lambda x: x / torch.max(torch.abs(x)))
    )
    
    # Create the data loader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=32,
        duration=5.0,  # 5 second clips
        transform=transform
    )
    
    # Example of iterating through the data
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        # Your training code would go here
        break 