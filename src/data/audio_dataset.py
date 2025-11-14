"""
Generic audio dataset implementation for Eta-WavLM training
Configurable via YAML files for different datasets and use cases.

Usage Examples:

# Create dataset from config file
dataset = AudioDataset(config_path="librispeech.yaml", split="train")

# Or create with config dict
config = load_config("librispeech.yaml")
dataset = AudioDataset(config=config, split="val")

# Create DataLoader with config
dataloader = create_dataloader("librispeech.yaml", split="train")

# Create speaker subset for classification
subset = dataset.create_speaker_subset()  # Uses speaker_subset config
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import re

import torch
import torchaudio
from torch.utils.data import Dataset
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class AudioDataset(Dataset):
    """
    Generic PyTorch Dataset for audio data
    
    Configurable via YAML files to work with different datasets:
    - LibriSpeech
    - VCTK
    - LJSpeech  
    - Custom datasets
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        split: str = "train",
        override_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audio dataset
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (alternative to config_path)
            split: Dataset split ("train", "val", "test")
            override_params: Parameters to override from config
        """
        # Load configuration
        if config_path is not None:
            self.config = load_config(config_path)
        elif config is not None:
            self.config = config.copy()
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Apply overrides
        if override_params:
            self.config.update(override_params)
        
        self.split = split
        self._setup_parameters()
        
        # Build dataset
        self.data_list = self._build_data_list()
        self._filter_and_limit_data()
        
        if hasattr(self, 'cache_speakers') and self.cache_speakers:
            self.speaker_info = self._build_speaker_info()
        
        logger.info(f"Dataset {split}: {len(self.data_list)} samples from {len(self.get_speaker_ids())} speakers")
    
    def _setup_parameters(self):
        """Extract parameters from config"""
        # Paths
        self.data_dir = Path(self.config['data_dir'])
        self.cache_dir = Path(self.config.get('cache_dir', './cache/data'))
        
        # Audio parameters
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.max_duration = self.config.get('max_duration', 10.0)
        self.min_duration = self.config.get('min_duration', 1.0)
        self.normalize = self.config.get('normalize', True)
        self.trim_silence = self.config.get('trim_silence', False)
        self.target_db = self.config.get('target_db', -20.0)
        
        # Data limits based on split
        limits_map = {
            'train': self.config.get('max_train_samples'),
            'val': self.config.get('max_val_samples'),
            'test': self.config.get('max_test_samples')
        }
        self.max_samples = limits_map.get(self.split)
        
        # Caching
        self.cache_speakers = True  # Always cache for efficiency
        
        # Split configuration
        split_map = {
            'train': self.config.get('train_split', 'train'),
            'val': self.config.get('val_split', 'val'),
            'test': self.config.get('test_split', 'test')
        }
        self.dataset_split = split_map[self.split]
        
        # File patterns (can be overridden for different datasets)
        self.file_patterns = self.config.get('file_patterns', {
            'audio_extensions': ['.wav', '.flac', '.mp3'],
            'speaker_id_pattern': r'(\d+)',  # Extract first number as speaker ID
            'recursive': True
        })
    
    def _build_data_list(self) -> List[Dict[str, Any]]:
        """Build list of audio files with metadata"""
        data_list = []
        
        # Determine search directory
        if self.dataset_split in ['train', 'val'] and self.config.get('val_split') == self.config.get('train_split'):
            # Using same split for train/val - will split later
            search_split = self.config.get('train_split', 'train')
        else:
            search_split = self.dataset_split
        
        search_dir = self._get_split_directory(search_split)
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {search_dir}")
        
        # Find audio files
        audio_files = self._find_audio_files(search_dir)
        
        for audio_path in audio_files:
            try:
                metadata = self._extract_metadata(audio_path)
                
                if metadata:
                    # Get audio info for duration filtering
                    try:
                        # Simple soundfile approach - most reliable
                        import soundfile as sf
                        info = sf.info(str(audio_path))
                        duration = info.duration
                        
                        # Filter by duration
                        if self.min_duration <= duration <= self.max_duration:
                            metadata.update({
                                'audio_path': str(audio_path),
                                'duration': duration,
                                'original_sr': info.samplerate,
                                'num_frames': info.frames,
                                'num_channels': info.channels
                            })
                            data_list.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not get info for {audio_path}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing {audio_path}: {e}")
                continue
        
        if not data_list:
            raise ValueError(f"No valid audio files found in {search_dir}")
        
        return data_list
    
    def _get_split_directory(self, split_name: str) -> Path:
        """Get directory for a specific split"""
        # Try common directory structures
        possible_paths = [
            self.data_dir / split_name,  # ./data/train
            self.data_dir / "LibriSpeech" / split_name,  # ./data/LibriSpeech/train-clean-100
            self.data_dir / split_name.replace('-', '_'),  # Handle different naming
            self.data_dir  # Single directory with all files
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to data_dir if nothing found
        return self.data_dir
    
    def _find_audio_files(self, search_dir: Path) -> List[Path]:
        """Find all audio files in directory"""
        audio_files = []
        extensions = self.file_patterns['audio_extensions']
        recursive = self.file_patterns.get('recursive', True)
        
        if recursive:
            for ext in extensions:
                audio_files.extend(search_dir.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                audio_files.extend(search_dir.glob(f"*{ext}"))
        
        return audio_files
    
    def _extract_metadata(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from audio file path"""
        try:
            # Extract speaker ID using pattern
            speaker_pattern = self.file_patterns.get('speaker_id_pattern', r'(\d+)')
            
            # Try to match speaker ID from filename or parent directories
            filename = audio_path.stem
            parent_names = [p.name for p in audio_path.parents]
            
            speaker_id = None
            
            # First try filename
            match = re.search(speaker_pattern, filename)
            if match:
                speaker_id = int(match.group(1))
            else:
                # Try parent directories
                for parent_name in parent_names:
                    match = re.search(speaker_pattern, parent_name)
                    if match:
                        speaker_id = int(match.group(1))
                        break
            
            if speaker_id is None:
                logger.warning(f"Could not extract speaker ID from {audio_path}")
                return None
            
            # Extract other metadata based on common patterns
            metadata = {
                'speaker_id': speaker_id,
                'filename': filename,
            }
            
            # Try to extract chapter/utterance info (LibriSpeech format)
            if '-' in filename:
                parts = filename.split('-')
                if len(parts) >= 3:
                    metadata.update({
                        'chapter_id': int(parts[1]) if parts[1].isdigit() else 0,
                        'utterance_id': '-'.join(parts[2:])
                    })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {audio_path}: {e}")
            return None
    
    def _filter_and_limit_data(self):
        """Apply train/val split and sample limits"""
        # Handle train/val split if using same source
        if (self.split in ['train', 'val'] and 
            self.config.get('val_split') == self.config.get('train_split')):
            
            train_data, val_data = self._create_train_val_split()
            
            if self.split == 'train':
                self.data_list = train_data
            else:
                self.data_list = val_data
        
        # Apply sample limits
        if self.max_samples and len(self.data_list) > self.max_samples:
            self.data_list = self.data_list[:self.max_samples]
    
    def _create_train_val_split(self) -> Tuple[List[Dict], List[Dict]]:
        """Create train/validation split"""
        val_ratio = self.config.get('val_ratio', 0.1)
        speaker_level = self.config.get('speaker_level_split', True)
        
        if speaker_level:
            # Split at speaker level
            speakers = list(set(item['speaker_id'] for item in self.data_list))
            n_val_speakers = int(len(speakers) * val_ratio)
            
            torch.manual_seed(42)  # For reproducibility
            perm = torch.randperm(len(speakers))
            val_speakers = set(speakers[i] for i in perm[:n_val_speakers])
            
            train_data = [item for item in self.data_list if item['speaker_id'] not in val_speakers]
            val_data = [item for item in self.data_list if item['speaker_id'] in val_speakers]
        else:
            # Split at utterance level
            n_val = int(len(self.data_list) * val_ratio)
            torch.manual_seed(42)
            perm = torch.randperm(len(self.data_list))
            
            val_indices = set(perm[:n_val].tolist())
            train_data = [item for i, item in enumerate(self.data_list) if i not in val_indices]
            val_data = [item for i, item in enumerate(self.data_list) if i in val_indices]
        
        return train_data, val_data
    
    def _build_speaker_info(self) -> Dict[int, Dict[str, Any]]:
        """Build speaker metadata dictionary"""
        speaker_info = {}
        
        for item in self.data_list:
            speaker_id = item['speaker_id']
            
            if speaker_id not in speaker_info:
                speaker_info[speaker_id] = {
                    'speaker_id': speaker_id,
                    'utterances': [],
                    'total_duration': 0.0,
                    'chapters': set()
                }
            
            speaker_info[speaker_id]['utterances'].append(item)
            speaker_info[speaker_id]['total_duration'] += item['duration']
            
            if 'chapter_id' in item:
                speaker_info[speaker_id]['chapters'].add(item['chapter_id'])
        
        # Convert sets to lists
        for speaker_id in speaker_info:
            if 'chapters' in speaker_info[speaker_id]:
                speaker_info[speaker_id]['chapters'] = list(speaker_info[speaker_id]['chapters'])
            speaker_info[speaker_id]['num_utterances'] = len(speaker_info[speaker_id]['utterances'])
        
        return speaker_info
    
    def get_speaker_ids(self) -> List[int]:
        """Get list of unique speaker IDs"""
        return list(set(item['speaker_id'] for item in self.data_list))
    
    def get_speaker_utterances(self, speaker_id: int) -> List[Dict[str, Any]]:
        """Get all utterances for a specific speaker"""
        return [item for item in self.data_list if item['speaker_id'] == speaker_id]
    
    def filter_by_speakers(self, speaker_ids: List[int]) -> 'AudioDataset':
        """Create new dataset filtered to specific speakers"""
        filtered_data = [item for item in self.data_list if item['speaker_id'] in speaker_ids]
        
        # Create new dataset with filtered data
        new_dataset = AudioDataset.__new__(AudioDataset)
        new_dataset.config = self.config.copy()
        new_dataset.split = f"{self.split}_filtered"
        new_dataset._setup_parameters()
        new_dataset.data_list = filtered_data
        
        return new_dataset
    
    def create_speaker_subset(self) -> 'AudioDataset':
        """Create subset for speaker classification based on config"""
        subset_config = self.config.get('speaker_subset', {})
        
        n_speakers = subset_config.get('n_speakers', 10)
        min_utterances = subset_config.get('min_utterances_per_speaker', 20)
        max_utterances = subset_config.get('max_utterances_per_speaker')
        
        # Filter speakers by utterance count
        speaker_counts = {}
        for item in self.data_list:
            speaker_id = item['speaker_id']
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        valid_speakers = [
            sid for sid, count in speaker_counts.items() 
            if count >= min_utterances
        ]
        
        # Select subset of speakers
        if len(valid_speakers) > n_speakers:
            torch.manual_seed(42)
            indices = torch.randperm(len(valid_speakers))[:n_speakers]
            selected_speakers = [valid_speakers[i] for i in indices]
        else:
            selected_speakers = valid_speakers
        
        # Filter data and optionally limit utterances per speaker
        filtered_data = []
        for speaker_id in selected_speakers:
            speaker_utterances = self.get_speaker_utterances(speaker_id)
            
            if max_utterances and len(speaker_utterances) > max_utterances:
                torch.manual_seed(42)
                indices = torch.randperm(len(speaker_utterances))[:max_utterances]
                speaker_utterances = [speaker_utterances[i] for i in indices]
            
            filtered_data.extend(speaker_utterances)
        
        # Create subset dataset
        subset_dataset = self.filter_by_speakers(selected_speakers)
        subset_dataset.data_list = filtered_data
        
        return subset_dataset
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample"""
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")
        
        item = self.data_list[idx]
        
        try:
            # Use soundfile directly for loading
            import soundfile as sf
            data, sr = sf.read(item['audio_path'])
    
            # Convert to torch tensor
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)  # soundfile uses (frames, channels)
    
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Apply preprocessing
            if self.trim_silence:
                waveform = self._trim_silence(waveform)
            
            if self.normalize:
                waveform = self._normalize_audio(waveform)
            
            return {
                'waveform': waveform,
                'sample_rate': self.sample_rate,
                'speaker_id': item['speaker_id'],
                'duration': item['duration'],
                'audio_path': item['audio_path'],
                **{k: v for k, v in item.items() if k not in ['audio_path', 'duration']}
            }
            
        except Exception as e:
            logger.error(f"Error loading audio {item['audio_path']}: {e}")
            # Return dummy sample
            dummy_length = int(self.sample_rate * 1.0)
            return {
                'waveform': torch.zeros(1, dummy_length),
                'sample_rate': self.sample_rate,
                'speaker_id': item['speaker_id'],
                'duration': 1.0,
                'audio_path': item['audio_path']
            }
    
    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim silence from beginning and end"""
        # Simple energy-based trimming
        energy = torch.sum(waveform ** 2, dim=0)
        threshold = 0.01 * torch.max(energy)
        
        # Find start and end of non-silent regions
        non_silent = energy > threshold
        if non_silent.any():
            start = torch.argmax(non_silent.float())
            end = len(non_silent) - torch.argmax(non_silent.flip(0).float())
            waveform = waveform[:, start:end]
        
        return waveform
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio amplitude"""
        if self.config.get('volume_normalize', False):
            # RMS normalization to target dB
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                target_rms = 10 ** (self.target_db / 20)
                waveform = waveform * (target_rms / rms)
        else:
            # Simple amplitude normalization
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
        
        return waveform
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if hasattr(self, 'speaker_info'):
            speaker_info = self.speaker_info
        else:
            speaker_info = self._build_speaker_info()
        
        durations = [item['duration'] for item in self.data_list]
        
        return {
            'num_samples': len(self.data_list),
            'num_speakers': len(speaker_info),
            'total_duration': sum(durations),
            'mean_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'sample_rate': self.sample_rate,
            'split': self.split,
            'dataset_split': self.dataset_split
        }


def create_dataloader(
    config_path: str,
    split: str = "train",
    shuffle: Optional[bool] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with configuration
    
    Args:
        config_path: Path to YAML config
        split: Dataset split
        shuffle: Whether to shuffle (auto-determined if None)
        **kwargs: Override DataLoader parameters
    """
    config = load_config(config_path)
    
    # Create dataset
    dataset = AudioDataset(config=config, split=split)
    
    # DataLoader parameters from config
    batch_size_key = f'{split}_batch_size' if f'{split}_batch_size' in config else 'batch_size'
    if split == 'train':
        batch_size_key = 'batch_size'
    elif split in ['val', 'test']:
        batch_size_key = 'eval_batch_size'
    
    loader_params = {
        'batch_size': config.get(batch_size_key, config.get('batch_size', 16)),
        'shuffle': shuffle if shuffle is not None else (split == 'train'),
        'num_workers': config.get('num_workers', 4),
        'pin_memory': config.get('pin_memory', True),
        'drop_last': config.get('drop_last', False),
        'collate_fn': collate_fn
    }
    
    if config.get('persistent_workers', True) and loader_params['num_workers'] > 0:
        loader_params['persistent_workers'] = True
    
    # Apply overrides
    loader_params.update(kwargs)
    
    return torch.utils.data.DataLoader(dataset, **loader_params)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for audio data
    Handles variable-length audio by padding to max length in batch.
    """
    # Extract components
    waveforms = [item['waveform'] for item in batch]
    sample_rates = [item['sample_rate'] for item in batch]
    speaker_ids = torch.tensor([item['speaker_id'] for item in batch])
    durations = torch.tensor([item['duration'] for item in batch])
    
    # Pad waveforms to same length
    max_length = max(w.shape[1] for w in waveforms)
    
    padded_waveforms = []
    lengths = []
    
    for waveform in waveforms:
        length = waveform.shape[1]
        lengths.append(length)
        
        if length < max_length:
            # Pad with zeros
            padding = max_length - length
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            padded_waveform = waveform
        
        padded_waveforms.append(padded_waveform)
    
    # Stack into batch
    waveforms_batch = torch.stack(padded_waveforms, dim=0)  # (batch_size, 1, max_length)
    lengths_batch = torch.tensor(lengths)
    
    # Collect other metadata
    metadata = {}
    for key in batch[0].keys():
        if key not in ['waveform', 'sample_rate', 'speaker_id', 'duration']:
            metadata[key] = [item[key] for item in batch]
    
    return {
        'waveforms': waveforms_batch,
        'lengths': lengths_batch,
        'sample_rate': sample_rates[0],  # Should be same for all
        'speaker_ids': speaker_ids,
        'durations': durations,
        **metadata
    }


if __name__ == "__main__":
    # Test the dataset
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a test config
    test_config = {
        'data_dir': './data/librispeech',
        'train_split': 'test-clean',
        'val_split': 'test-clean', 
        'test_split': 'test-clean',
        'sample_rate': 16000,
        'max_duration': 10.0,
        'min_duration': 1.0,
        'max_test_samples': 50,
        'batch_size': 4,
        'num_workers': 2
    }
    
    try:
        # Test dataset creation
        dataset = AudioDataset(config=test_config, split="test")
        print(f"Dataset size: {len(dataset)}")
        print(f"Stats: {dataset.get_dataset_stats()}")
        
        # Test sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shape: {sample['waveform'].shape}")
            print(f"Speaker ID: {sample['speaker_id']}")
        
        # Test dataloader  
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=2,
            collate_fn=collate_fn,
            shuffle=True
        )
        
        batch = next(iter(dataloader))
        print(f"Batch waveforms shape: {batch['waveforms'].shape}")
        print(f"Batch speaker IDs: {batch['speaker_ids']}")
        
    except Exception as e:
        print(f"Test failed (expected if no data): {e}")
        print("This is normal if you don't have LibriSpeech data downloaded")
