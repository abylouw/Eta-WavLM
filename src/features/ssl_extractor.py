"""
SSL Feature Extractors for Eta-WavLM

This module provides a flexible interface for extracting features from different
self-supervised speech models (WavLM, Wav2Vec2, HuBERT) using a factory pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Tuple
import logging
import warnings
import re

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor, 
    WavLMModel, 
    Wav2Vec2Model,
    HubertModel,
    WavLMConfig,
    Wav2Vec2Config,
    HubertConfig
)
import numpy as np

logger = logging.getLogger(__name__)

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class BaseSSLExtractor(nn.Module, ABC):
    """
    Abstract base class for SSL feature extractors
    
    Provides common interface for different SSL models while allowing
    model-specific implementations.
    """
    
    def __init__(
        self,
        model_name: str,
        layer_index: int,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        freeze_model: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize base SSL feature extractor
        
        Args:
            model_name: HuggingFace model name
            layer_index: Which transformer layer to extract (0-indexed)
            sample_rate: Expected audio sample rate
            device: Device to run model on
            freeze_model: Whether to freeze model parameters
            cache_dir: Cache directory for model files
        """
        super().__init__()
        
        self.model_name = model_name
        self.layer_index = layer_index
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_model = freeze_model
        self.cache_dir = cache_dir
        
        # Will be set by subclasses
        self.model = None
        self.feature_extractor = None
        self.hidden_size = None
        self.num_layers = None
        
        # Initialize model-specific components
        self._load_model()
        self._setup_model()
        
        # Validate layer index
        if self.layer_index >= self.num_layers:
            raise ValueError(f"Layer index {self.layer_index} >= num layers {self.num_layers}")
        
        logger.info(f"Initialized {self.__class__.__name__} with {self.model_name}, "
                   f"layer {self.layer_index + 1}/{self.num_layers}, hidden size {self.hidden_size}")
    
    @abstractmethod
    def _load_model(self):
        """Load the specific SSL model and feature extractor"""
        pass
    
    def _setup_model(self):
        """Common model setup (move to device, freeze, etc.)"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Freeze model if specified
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get model dimensions
        config = self.model.config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
    
    def preprocess_audio(self, waveforms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Preprocess audio for model input
        
        Args:
            waveforms: Audio waveforms (batch_size, channels, samples) or (batch_size, samples)
            
        Returns:
            Dictionary with preprocessed inputs
        """
        # Handle different input shapes
        if waveforms.dim() == 3:
            # (batch_size, channels, samples) -> (batch_size, samples)
            waveforms = waveforms.squeeze(1)
        elif waveforms.dim() == 1:
            # (samples,) -> (1, samples)
            waveforms = waveforms.unsqueeze(0)
        
        # Convert to numpy for feature extractor
        if isinstance(waveforms, torch.Tensor):
            waveforms_np = waveforms.cpu().numpy()
        else:
            waveforms_np = waveforms
        
        # Process each waveform in the batch
        batch_size = len(waveforms_np)
        processed_inputs = []
        
        for i in range(batch_size):
            # Feature extractor expects 1D numpy array
            waveform = waveforms_np[i]
            
            # Apply feature extraction
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            processed_inputs.append(inputs["input_values"].squeeze(0))
        
        # Stack into batch
        input_values = torch.stack(processed_inputs, dim=0)
        
        return {
            "input_values": input_values.to(self.device)
        }
    
    def extract_features(
        self, 
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Extract SSL features from model
        
        Args:
            waveforms: Input audio waveforms
            lengths: Actual lengths of each waveform (for masking)
            return_all_layers: Whether to return all layer outputs
            
        Returns:
            SSL features from specified layer (batch_size, seq_len, hidden_size)
            or all layers if return_all_layers=True
        """
        # Preprocess audio
        inputs = self.preprocess_audio(waveforms)
        
        # Extract features with appropriate gradient setting
        with torch.set_grad_enabled(not self.freeze_model):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        if return_all_layers:
            # Return all hidden states
            return outputs.hidden_states
        else:
            # Return specific layer
            layer_output = outputs.hidden_states[self.layer_index]
            return layer_output
    
    def forward(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - extract features from specified layer"""
        return self.extract_features(waveforms, lengths)
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension"""
        return self.hidden_size
    
    @abstractmethod
    def get_subsampling_factor(self) -> int:
        """Get the subsampling factor of the model"""
        pass
    
    def compute_sequence_length(self, audio_length: int) -> int:
        """
        Compute output sequence length given input audio length
        
        Args:
            audio_length: Input audio length in samples
            
        Returns:
            Output sequence length
        """
        return audio_length // self.get_subsampling_factor()
    
    @torch.no_grad()
    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Features as numpy array
        """
        import torchaudio
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract features
        features = self.extract_features(waveform.unsqueeze(0))
        
        return features.squeeze(0).cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": self.__class__.__name__,
            "model_name": self.model_name,
            "layer_index": self.layer_index,
            "layer_number": self.layer_index + 1,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "frozen": self.freeze_model,
            "subsampling_factor": self.get_subsampling_factor()
        }


class WavLMExtractor(BaseSSLExtractor):
    """WavLM feature extractor implementation"""
    
    def _load_model(self):
        """Load WavLM model and feature extractor"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = WavLMModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
    
    def get_subsampling_factor(self) -> int:
        """WavLM subsampling factor"""
        return 320  # 20ms frames at 16kHz


class Wav2Vec2Extractor(BaseSSLExtractor):
    """Wav2Vec2 feature extractor implementation"""
    
    def _load_model(self):
        """Load Wav2Vec2 model and feature extractor"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = Wav2Vec2Model.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
    
    def get_subsampling_factor(self) -> int:
        """Wav2Vec2 subsampling factor"""
        return 320  # 20ms frames at 16kHz


class HuBERTExtractor(BaseSSLExtractor):
    """HuBERT feature extractor implementation"""
    
    def _load_model(self):
        """Load HuBERT model and feature extractor"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = HubertModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
    
    def get_subsampling_factor(self) -> int:
        """HuBERT subsampling factor"""
        return 320  # 20ms frames at 16kHz


class FrameSampler:
    """
    Utility class for sampling frames from SSL features
    
    As mentioned in the paper, they randomly subsample L frames from each utterance
    during training to create fixed-length representations.
    """
    
    def __init__(self, num_frames: int = 100, sampling_strategy: str = "random", random_seed: Optional[int] = None):
        """
        Initialize frame sampler
        
        Args:
            num_frames: Number of frames to sample (L in paper)
            sampling_strategy: Sampling strategy ("random" or "uniform")
            random_seed: Random seed for reproducibility
        """
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.random_seed = random_seed
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        if sampling_strategy not in ["random", "uniform"]:
            raise ValueError(f"Sampling strategy must be 'random' or 'uniform', got {sampling_strategy}")
    
    def sample_frames(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sample frames from feature sequence
        
        Args:
            features: Input features (seq_len, hidden_size) or (batch_size, seq_len, hidden_size)
            
        Returns:
            Sampled features (num_frames, hidden_size) or (batch_size, num_frames, hidden_size)
        """
        if features.dim() == 2:
            # Single sequence
            return self._sample_single_sequence(features)
        elif features.dim() == 3:
            # Batch of sequences
            batch_size = features.shape[0]
            sampled_batch = []
            
            for i in range(batch_size):
                sampled_features = self._sample_single_sequence(features[i])
                sampled_batch.append(sampled_features)
            
            return torch.stack(sampled_batch, dim=0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {features.dim()}D")
    
    def _sample_single_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """Sample frames from a single sequence"""
        seq_len, hidden_size = features.shape
        
        if seq_len <= self.num_frames:
            # Pad if too short
            padding = self.num_frames - seq_len
            padded_features = torch.nn.functional.pad(features, (0, 0, 0, padding))
            return padded_features
        
        if self.sampling_strategy == "random":
            # Randomly sample frames
            indices = torch.randperm(seq_len)[:self.num_frames]
            indices = torch.sort(indices)[0]  # Sort to maintain temporal order
            return features[indices]
        
        elif self.sampling_strategy == "uniform":
            # Uniform sampling
            step = seq_len / self.num_frames
            indices = torch.arange(0, seq_len, step, dtype=torch.long)[:self.num_frames]
            return features[indices]


def create_ssl_extractor(config: Dict[str, Any]) -> BaseSSLExtractor:
    """
    Factory function to create SSL feature extractor based on config
    
    Args:
        config: Configuration dictionary containing ssl_model parameters
        
    Returns:
        Appropriate SSL feature extractor instance
    """
    ssl_config = config.get('ssl_model', {})
    model_name = ssl_config.get('name', 'microsoft/wavlm-large')
    layer_index = ssl_config.get('layer_index', 14)
    freeze = ssl_config.get('freeze', True)
    cache_dir = ssl_config.get('cache_dir', './cache/models/ssl')
    
    # Determine model type from name
    model_name_lower = model_name.lower()
    
    if 'wavlm' in model_name_lower:
        extractor_class = WavLMExtractor
    elif 'wav2vec2' in model_name_lower:
        extractor_class = Wav2Vec2Extractor
    elif 'hubert' in model_name_lower:
        extractor_class = HuBERTExtractor
    else:
        raise ValueError(f"Unsupported model type in name: {model_name}. "
                        f"Supported types: wavlm, wav2vec2, hubert")
    
    # Create extractor
    extractor = extractor_class(
        model_name=model_name,
        layer_index=layer_index,
        freeze_model=freeze,
        cache_dir=cache_dir
    )
    
    logger.info(f"Created {extractor.__class__.__name__} for {model_name}")
    
    return extractor


def create_frame_sampler(config: Dict[str, Any]) -> FrameSampler:
    """
    Factory function to create frame sampler based on config
    
    Args:
        config: Configuration dictionary containing feature_extraction parameters
        
    Returns:
        Configured FrameSampler instance
    """
    feature_config = config.get('feature_extraction', {})
    
    return FrameSampler(
        num_frames=feature_config.get('num_frames', 100),
        sampling_strategy=feature_config.get('frame_sampling', 'random'),
        random_seed=42  # For reproducibility
    )


if __name__ == "__main__":
    # Test the SSL feature extractors
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test config
    test_config = {
        'ssl_model': {
            'name': 'microsoft/wavlm-large',
            'layer_index': 14,
            'freeze': True,
            'cache_dir': './cache/models/ssl'
        },
        'feature_extraction': {
            'num_frames': 100,
            'frame_sampling': 'random'
        }
    }
    
    try:
        # Create extractor
        extractor = create_ssl_extractor(test_config)
        print(f"Model info: {extractor.get_model_info()}")
        
        # Create frame sampler
        sampler = create_frame_sampler(test_config)
        
        # Test with dummy audio
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        dummy_audio = torch.randn(batch_size, 1, audio_length)
        
        # Extract features
        features = extractor(dummy_audio)
        print(f"Input audio shape: {dummy_audio.shape}")
        print(f"Output features shape: {features.shape}")
        print(f"Feature dimension: {extractor.get_output_dim()}")
        
        # Test frame sampling
        sampled_features = sampler.sample_frames(features)
        print(f"Sampled features shape: {sampled_features.shape}")
        
        # Test different model types
        for model_name in ['microsoft/wavlm-base', 'facebook/wav2vec2-base-960h', 'facebook/hubert-base-ls960']:
            try:
                test_config['ssl_model']['name'] = model_name
                test_extractor = create_ssl_extractor(test_config)
                print(f"Successfully created extractor for {model_name}: {test_extractor.__class__.__name__}")
            except Exception as e:
                print(f"Failed to create extractor for {model_name}: {e}")
                
    except Exception as e:
        print(f"Test failed (expected if models not available): {e}")
