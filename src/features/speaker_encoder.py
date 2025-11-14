"""
Speaker Encoders for Eta-WavLM

This module provides a flexible interface for extracting speaker embeddings from different
speaker recognition models (ECAPA-TDNN, Resemblyzer, WavLM-SV) using a factory pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import logging
import warnings

import torch
import torch.nn as nn
import numpy as np
import torchaudio

logger = logging.getLogger(__name__)

# Suppress various warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
#warnings.filterwarnings("ignore", category=UserWarning, module="resemblyzer")


class BaseSpeakerEncoder(nn.Module, ABC):
    """
    Abstract base class for speaker encoders
    
    Provides common interface for different speaker recognition models while allowing
    model-specific implementations.
    """
    
    def __init__(
        self,
        model_source: str,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        freeze_model: bool = True
    ):
        """
        Initialize base speaker encoder
        
        Args:
            model_source: Model source/path identifier
            sample_rate: Expected audio sample rate
            device: Device to run model on
            cache_dir: Cache directory for model files
            freeze_model: Whether to freeze model parameters
        """
        super().__init__()
        
        self.model_source = model_source
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.freeze_model = freeze_model
        
        # Will be set by subclasses
        self.model = None
        self.embedding_dim = None
        
        # Initialize model-specific components
        self._load_model()
        self._setup_model()
        
        logger.info(f"Initialized {self.__class__.__name__} with {self.model_source}, "
                   f"embedding dim: {self.embedding_dim}")
    
    @abstractmethod
    def _load_model(self):
        """Load the specific speaker recognition model"""
        pass
    
    def _setup_model(self):
        """Common model setup (freeze, eval mode, etc.)"""
        if self.freeze_model and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, 'eval'):
                self.model.eval()
    
    def preprocess_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for model input
        
        Args:
            waveforms: Audio waveforms (batch_size, channels, samples) or (batch_size, samples)
            
        Returns:
            Preprocessed waveforms
        """
        # Handle different input shapes
        if waveforms.dim() == 3:
            # (batch_size, channels, samples) -> (batch_size, samples)
            waveforms = waveforms.squeeze(1)
        elif waveforms.dim() == 1:
            # (samples,) -> (1, samples)
            waveforms = waveforms.unsqueeze(0)
        elif waveforms.dim() == 2:
            # Already correct shape (batch_size, samples)
            pass
        else:
            raise ValueError(f"Unexpected waveform shape: {waveforms.shape}")
        
        # Ensure tensor is on correct device
        waveforms = waveforms.to(self.device)
        
        return waveforms
    
    @abstractmethod
    def extract_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract speaker embeddings from audio
        
        Args:
            waveforms: Input audio waveforms (batch_size, samples)
            lengths: Actual lengths of each waveform (optional)
            
        Returns:
            Speaker embeddings (batch_size, embedding_dim)
        """
        pass
    
    def forward(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - extract speaker embeddings"""
        return self.extract_embeddings(waveforms, lengths)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    @torch.no_grad()
    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding as numpy array
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract embedding
        embedding = self.extract_embeddings(waveform.unsqueeze(0))
        
        return embedding.squeeze(0).cpu().numpy()
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between speaker embeddings
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
        embedding2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(embedding1_norm * embedding2_norm, dim=-1)
        
        return similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": self.__class__.__name__,
            "model_source": self.model_source,
            "embedding_dim": self.embedding_dim,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "frozen": self.freeze_model
        }


class ECAPATDNNEncoder(BaseSpeakerEncoder):
    """ECAPA-TDNN speaker encoder using SpeechBrain"""
    
    def _load_model(self):
        """Load ECAPA-TDNN model from SpeechBrain"""
        # Get the specific logger instance for SpeechBrain
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError:
                raise ImportError(
                    "Could not import EncoderClassifier from SpeechBrain. "
                    "Please install speechbrain: pip install speechbrain"
                )
        
        try:
            self.model = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts={"device": self.device},
                savedir=self.cache_dir
            )
            
            # Get embedding dimension
            self.embedding_dim = self._get_embedding_dim()
            
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model {self.model_source}: {e}")
            raise
    
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by running test input"""
        try:
            dummy_audio = torch.randn(1, self.sample_rate).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_batch(dummy_audio)
            return embedding.shape[-1]
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return 192  # Default ECAPA-TDNN size
    
    def _setup_model(self):
        """Setup ECAPA-TDNN specific configuration"""
        super()._setup_model()
        
        # Freeze SpeechBrain model components
        if self.freeze_model and hasattr(self.model, 'mods'):
            for param in self.model.mods.parameters():
                param.requires_grad = False
    
    def extract_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract ECAPA-TDNN embeddings"""
        processed_audio = self.preprocess_audio(waveforms)
        
        # Ensure SpeechBrain model is on correct device
        self.model.device = self.device
        if hasattr(self.model, 'mods'):
            for name, module in self.model.mods.items():
                self.model.mods[name] = module.to(self.device)
        
        with torch.set_grad_enabled(not self.freeze_model):
            try:
                embeddings = self.model.encode_batch(processed_audio)
                
                # Handle different output formats
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                
                # Ensure correct shape
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(1)
                
                return embeddings
                
            except Exception as e:
                logger.error(f"Error extracting ECAPA-TDNN embeddings: {e}")
                batch_size = processed_audio.shape[0]
                return torch.zeros(batch_size, self.embedding_dim, device=self.device)


class ResemblyzerEncoder(BaseSpeakerEncoder):
    """Resemblyzer speaker encoder using GE2E-based embeddings"""
    
    def _load_model(self):
        """Load Resemblyzer model"""
        try:
            from resemblyzer import VoiceEncoder
            self.model = VoiceEncoder(device=self.device)
            self.embedding_dim = 256  # Resemblyzer embedding dimension
            
        except ImportError:
            raise ImportError(
                "Could not import VoiceEncoder from resemblyzer. "
                "Please install resemblyzer: pip install resemblyzer"
            )
        except Exception as e:
            logger.error(f"Failed to load Resemblyzer model: {e}")
            raise
    
    def extract_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract Resemblyzer embeddings"""
        processed_audio = self.preprocess_audio(waveforms)
        
        try:
            # Convert to numpy (Resemblyzer expects numpy)
            audio_np = processed_audio.cpu().numpy()
            
            # Extract embeddings for each sample in batch
            embeddings = []
            for i in range(audio_np.shape[0]):
                # Resemblyzer expects 1D audio at 16kHz
                embedding = self.model.embed_utterance(audio_np[i])
                embeddings.append(embedding)
            
            # Stack and convert back to torch tensor
            embeddings = np.stack(embeddings, axis=0)
            embeddings = torch.from_numpy(embeddings).to(self.device)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting Resemblyzer embeddings: {e}")
            batch_size = processed_audio.shape[0]
            return torch.zeros(batch_size, self.embedding_dim, device=self.device)


class WavLMSVEncoder(BaseSpeakerEncoder):
    """WavLM Speaker Verification encoder"""
    
    def _load_model(self):
        """Load WavLM-SV model"""
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_source,
                cache_dir=self.cache_dir
            )
            
            self.model = WavLMForXVector.from_pretrained(
                self.model_source,
                cache_dir=self.cache_dir
            )
            
            self.model = self.model.to(self.device)
            self.embedding_dim = self.model.config.tdnn_dim[-1]  # Last TDNN layer dimension
            
        except ImportError:
            raise ImportError(
                "Could not import WavLMForXVector. Please ensure you have the correct transformers version."
            )
        except Exception as e:
            logger.error(f"Failed to load WavLM-SV model {self.model_source}: {e}")
            raise
    
    def extract_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract WavLM-SV embeddings"""
        processed_audio = self.preprocess_audio(waveforms)
        
        try:
            # Preprocess with feature extractor
            audio_np = processed_audio.cpu().numpy()
            inputs = self.feature_extractor(
                audio_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.set_grad_enabled(not self.freeze_model):
                outputs = self.model(**inputs)
                embeddings = outputs.embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting WavLM-SV embeddings: {e}")
            batch_size = processed_audio.shape[0]
            return torch.zeros(batch_size, self.embedding_dim, device=self.device)


class XVectorEncoder(BaseSpeakerEncoder):
    """X-Vector speaker encoder (placeholder for future implementation)"""
    
    def _load_model(self):
        """Load X-Vector model"""
        # Placeholder - would implement actual X-Vector loading
        raise NotImplementedError("X-Vector encoder not yet implemented")
    
    def extract_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract X-Vector embeddings"""
        raise NotImplementedError("X-Vector encoder not yet implemented")


class SpeakerEmbeddingReplicator:
    """
    Utility class for replicating speaker embeddings across time frames
    
    In the Eta-WavLM paper, speaker embeddings are replicated L times along the 
    frame axis to match the SSL feature sequence length for training the linear
    decomposition.
    """
    
    def __init__(self, num_frames: int = 100):
        """
        Initialize embedding replicator
        
        Args:
            num_frames: Number of frames to replicate (L in paper)
        """
        self.num_frames = num_frames
    
    def replicate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Replicate speaker embeddings along the frame axis
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            
        Returns:
            Replicated embeddings (batch_size, num_frames, embedding_dim)
        """
        batch_size, embedding_dim = embeddings.shape
        
        # Replicate embeddings L times
        replicated = embeddings.unsqueeze(1).expand(batch_size, self.num_frames, embedding_dim)
        
        return replicated
    
    def flatten_replicated(self, replicated_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Flatten replicated embeddings for training
        
        Args:
            replicated_embeddings: Replicated embeddings (batch_size, num_frames, embedding_dim)
            
        Returns:
            Flattened embeddings (batch_size * num_frames, embedding_dim)
        """
        batch_size, num_frames, embedding_dim = replicated_embeddings.shape
        return replicated_embeddings.view(-1, embedding_dim)


def create_speaker_encoder(config: Dict[str, Any]) -> BaseSpeakerEncoder:
    """
    Factory function to create speaker encoder based on config
    
    Args:
        config: Configuration dictionary containing speaker_model parameters
        
    Returns:
        Appropriate speaker encoder instance
    """
    speaker_config = config.get('speaker_model', {})
    model_source = speaker_config.get('source', 'speechbrain/spkrec-ecapa-voxceleb')
    freeze = speaker_config.get('freeze', True)
    cache_dir = speaker_config.get('cache_dir', './cache/models/speaker')
    
    # Determine encoder type from model source
    model_source_lower = model_source.lower()
    
    if 'ecapa' in model_source_lower or 'speechbrain' in model_source_lower:
        encoder_class = ECAPATDNNEncoder
    elif 'resemblyzer' in model_source_lower:
        encoder_class = ResemblyzerEncoder
    elif 'wavlm' in model_source_lower and ('sv' in model_source_lower or 'speaker' in model_source_lower):
        encoder_class = WavLMSVEncoder
    elif 'xvector' in model_source_lower:
        encoder_class = XVectorEncoder
    else:
        # Default to ECAPA-TDNN for SpeechBrain models
        logger.warning(f"Could not determine encoder type from {model_source}, defaulting to ECAPA-TDNN")
        encoder_class = ECAPATDNNEncoder
    
    # Create encoder
    encoder = encoder_class(
        model_source=model_source,
        freeze_model=freeze,
        cache_dir=cache_dir
    )
    
    logger.info(f"Created {encoder.__class__.__name__} for {model_source}")
    
    return encoder


def create_speaker_replicator(config: Dict[str, Any]) -> SpeakerEmbeddingReplicator:
    """
    Factory function to create speaker embedding replicator based on config
    
    Args:
        config: Configuration dictionary containing feature_extraction parameters
        
    Returns:
        Configured SpeakerEmbeddingReplicator instance
    """
    feature_config = config.get('feature_extraction', {})
    num_frames = feature_config.get('num_frames', 100)
    
    return SpeakerEmbeddingReplicator(num_frames=num_frames)


def verify_speaker_encoder(model_source: str = "speechbrain/spkrec-ecapa-voxceleb") -> bool:
    """
    Verify that a speaker encoder can be loaded and used
    
    Args:
        model_source: Model source to verify
        
    Returns:
        True if verification successful, False otherwise
    """
    try:
        # Create test config
        test_config = {
            'speaker_model': {
                'source': model_source,
                'freeze': True,
                'cache_dir': './cache/models/speaker'
            }
        }
        
        # Test initialization
        encoder = create_speaker_encoder(test_config)
        
        # Test with dummy audio
        dummy_audio = torch.randn(2, 16000)  # 2 samples, 1 second each
        embeddings = encoder(dummy_audio)
        
        # Check output shape
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == encoder.get_embedding_dim()
        
        logger.info(f"Speaker encoder verification successful: {embeddings.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Speaker encoder verification failed: {e}")
        return False


if __name__ == "__main__":
    # Test the speaker encoders
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configs for different encoders
    test_configs = [
        {
            'speaker_model': {
                'source': 'speechbrain/spkrec-ecapa-voxceleb',
                'freeze': True,
                'cache_dir': './cache/models/speaker'
            },
            'feature_extraction': {
                'num_frames': 100
            }
        }
    ]
    
    for i, config in enumerate(test_configs):
        try:
            print(f"\n=== Testing config {i+1} ===")
            
            # Verify encoder
            model_source = config['speaker_model']['source']
            if not verify_speaker_encoder(model_source):
                print(f"Verification failed for {model_source}")
                continue
            
            # Create encoder and replicator
            encoder = create_speaker_encoder(config)
            replicator = create_speaker_replicator(config)
            
            print(f"Model info: {encoder.get_model_info()}")
            
            # Test with dummy audio
            batch_size = 2
            audio_length = 32000  # 2 seconds at 16kHz
            dummy_audio = torch.randn(batch_size, audio_length)
            
            # Extract embeddings
            embeddings = encoder(dummy_audio)
            print(f"Input audio shape: {dummy_audio.shape}")
            print(f"Output embeddings shape: {embeddings.shape}")
            
            # Test replication
            replicated = replicator.replicate_embeddings(embeddings)
            print(f"Replicated embeddings shape: {replicated.shape}")
            
            # Test similarity
            sim = encoder.compute_similarity(embeddings[0:1], embeddings[1:2])
            print(f"Similarity between speakers: {sim.item():.4f}")
            
        except Exception as e:
            print(f"Test failed for config {i+1}: {e}")
