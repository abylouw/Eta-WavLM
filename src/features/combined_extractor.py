"""
Combined Feature Extractor for Eta-WavLM

This module combines SSL feature extraction and speaker embedding extraction
using our modular extractors to prepare data for the linear decomposition training.
"""

from typing import Dict, Any, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

from .ssl_extractor import create_ssl_extractor, create_frame_sampler
from .speaker_encoder import create_speaker_encoder, create_speaker_replicator

logger = logging.getLogger(__name__)


class CombinedFeatureExtractor(nn.Module):
    """
    Combined Feature Extractor for Eta-WavLM Training
    
    Uses modular SSL extractors and speaker encoders to extract both SSL features (s) 
    and speaker embeddings (d) from audio, preparing them for the linear decomposition: s = f(d) + η
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Combined Feature Extractor from config
        
        Args:
            config: Configuration dictionary containing ssl_model, speaker_model, 
                   and feature_extraction parameters
        """
        super().__init__()
        
        self.config = config
        
        # Extract configuration parameters
        feature_config = config.get('feature_extraction', {})
        self.num_frames = feature_config.get('num_frames', 100)
        self.pca_components = feature_config.get('pca_components', 128)
        self.fit_pca_on_first_batch = feature_config.get('fit_pca_on_first_batch', True)
        
        # Create SSL feature extractor using factory
        self.ssl_extractor = create_ssl_extractor(config)
        
        # Create speaker encoder using factory  
        self.speaker_encoder = create_speaker_encoder(config)
        
        # Create utilities using factories
        self.frame_sampler = create_frame_sampler(config)
        self.speaker_replicator = create_speaker_replicator(config)
        
        # PCA for dimensionality reduction (will be fitted during training)
        self.pca = None
        self.pca_fitted = False
        
        # Get device from SSL extractor (they should be on same device)
        self.device = self.ssl_extractor.device
        
        logger.info(f"Initialized CombinedFeatureExtractor:")
        logger.info(f"  SSL: {self.ssl_extractor.get_model_info()['model_name']}")
        logger.info(f"  Speaker: {self.speaker_encoder.get_model_info()['model_source']}")
        logger.info(f"  Frames: {self.num_frames}, PCA components: {self.pca_components}")
    
    def extract_ssl_features(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract and sample SSL features
        
        Args:
            waveforms: Input audio waveforms
            lengths: Actual lengths of waveforms
            
        Returns:
            Sampled SSL features (batch_size, num_frames, hidden_size)
        """
        # Extract raw SSL features
        ssl_features = self.ssl_extractor(waveforms, lengths)
        
        # Sample frames to fixed length
        sampled_features = self.frame_sampler.sample_frames(ssl_features)
        
        return sampled_features
    
    def extract_speaker_embeddings(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract and replicate speaker embeddings
        
        Args:
            waveforms: Input audio waveforms  
            lengths: Actual lengths of waveforms
            
        Returns:
            Replicated speaker embeddings (batch_size, num_frames, embedding_dim)
        """
        # Extract speaker embeddings
        speaker_embeddings = self.speaker_encoder(waveforms, lengths)
        
        # Replicate across time frames
        replicated_embeddings = self.speaker_replicator.replicate_embeddings(speaker_embeddings)
        
        return replicated_embeddings
    
    def extract_features_for_training(
        self, 
        waveforms: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        fit_pca: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features formatted for linear decomposition training
        
        Args:
            waveforms: Input audio waveforms
            lengths: Actual lengths of waveforms
            fit_pca: Whether to fit PCA on this batch (typically first batch)
            
        Returns:
            Dictionary with 'ssl_features' and 'speaker_embeddings' formatted for training
        """
        # Extract SSL features (batch_size, num_frames, ssl_dim)
        ssl_features = self.extract_ssl_features(waveforms, lengths)
        
        # Extract speaker embeddings (batch_size, num_frames, speaker_dim)
        speaker_embeddings = self.extract_speaker_embeddings(waveforms, lengths)
        
        # Fit PCA on first batch if requested
        if fit_pca and not self.pca_fitted:
            # Flatten speaker embeddings for PCA fitting
            speaker_embeddings_flat = speaker_embeddings.reshape(-1, speaker_embeddings.shape[-1])
            self.fit_pca(speaker_embeddings_flat)
        
        # Apply PCA if fitted
        if self.pca_fitted:
            speaker_embeddings = self.apply_pca(speaker_embeddings)
        
        # Flatten for training: (batch_size * num_frames, dim)
        ssl_features_flat = ssl_features.reshape(-1, ssl_features.shape[-1])
        speaker_embeddings_flat = speaker_embeddings.reshape(-1, speaker_embeddings.shape[-1])
        
        return {
            'ssl_features': ssl_features_flat,  # (N, ssl_dim) - 'S' matrix in paper
            'speaker_embeddings': speaker_embeddings_flat,  # (N, speaker_dim) - 'D' matrix in paper
            'ssl_features_shaped': ssl_features,  # Keep original shapes for reference
            'speaker_embeddings_shaped': speaker_embeddings
        }
    
    def fit_pca(self, speaker_embeddings: torch.Tensor):
        """
        Fit PCA on speaker embeddings for dimensionality reduction
        
        Args:
            speaker_embeddings: Speaker embeddings to fit PCA on (N, embedding_dim)
        """
        if isinstance(speaker_embeddings, torch.Tensor):
            speaker_embeddings = speaker_embeddings.cpu().numpy()
        
        logger.info(f"Fitting PCA on {speaker_embeddings.shape[0]} speaker embeddings")
        logger.info(f"Original embedding dim: {speaker_embeddings.shape[1]} -> PCA dim: {self.pca_components}")
        
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        self.pca.fit(speaker_embeddings)
        self.pca_fitted = True
        
        # Log explained variance
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_var:.3f} with {self.pca_components} components")
        
        # Log top components
        top_components = self.pca.explained_variance_ratio_[:5]
        logger.info(f"Top 5 component variances: {top_components}")
    
    def apply_pca(self, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply fitted PCA to speaker embeddings
        
        Args:
            speaker_embeddings: Speaker embeddings (any shape with last dim = embedding_dim)
            
        Returns:
            PCA-transformed embeddings (same shape with last dim = pca_components)
        """
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before applying. Call fit_pca() first.")
        
        original_shape = speaker_embeddings.shape
        device = speaker_embeddings.device
        
        # Flatten to 2D for PCA
        flat_embeddings = speaker_embeddings.reshape(-1, original_shape[-1])
        
        # Apply PCA
        flat_embeddings_np = flat_embeddings.cpu().numpy()
        transformed_np = self.pca.transform(flat_embeddings_np)
        transformed = torch.from_numpy(transformed_np).to(device).float()
        
        # Reshape back to original shape (except last dimension)
        new_shape = original_shape[:-1] + (self.pca_components,)
        transformed = transformed.view(new_shape)
        
        return transformed
    
    def forward(
        self, 
        waveforms: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        fit_pca: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - extract features for training
        
        Args:
            waveforms: Input audio waveforms
            lengths: Actual lengths of waveforms
            fit_pca: Whether to fit PCA on this batch
            
        Returns:
            Features formatted for linear decomposition training
        """
        return self.extract_features_for_training(waveforms, lengths, fit_pca)
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions"""
        ssl_dim = self.ssl_extractor.get_output_dim()
        speaker_dim = self.speaker_encoder.get_embedding_dim()
        
        if self.pca_fitted:
            speaker_dim = self.pca_components
        
        return {
            'ssl_dim': ssl_dim,  # Q in paper
            'speaker_dim': speaker_dim,  # P in paper (after PCA)
            'num_frames': self.num_frames  # L in paper
        }
    
    def save_pca(self, path: str):
        """Save fitted PCA model"""
        if not self.pca_fitted:
            raise RuntimeError("PCA not fitted yet")
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.pca, f)
        logger.info(f"Saved PCA model to {path}")
    
    def load_pca(self, path: str):
        """Load fitted PCA model"""
        import pickle
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        self.pca_fitted = True
        logger.info(f"Loaded PCA model from {path}")
        logger.info(f"PCA components: {self.pca.n_components_}")
    
    def get_pca_info(self) -> Dict[str, Any]:
        """Get PCA information"""
        if not self.pca_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "n_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(np.sum(self.pca.explained_variance_ratio_))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all components"""
        ssl_info = self.ssl_extractor.get_model_info()
        speaker_info = self.speaker_encoder.get_model_info()
        
        return {
            'ssl': ssl_info,
            'speaker': speaker_info,
            'num_frames': self.num_frames,
            'pca_components': self.pca_components,
            'pca_info': self.get_pca_info(),
            'feature_dims': self.get_feature_dims(),
            'fit_pca_on_first_batch': self.fit_pca_on_first_batch
        }


def create_combined_extractor(config: Dict[str, Any]) -> CombinedFeatureExtractor:
    """
    Create CombinedFeatureExtractor from configuration
    
    Args:
        config: Full configuration dictionary with ssl_model, speaker_model, 
               and feature_extraction sections
        
    Returns:
        Configured CombinedFeatureExtractor
    """
    return CombinedFeatureExtractor(config)


def load_config_and_create_extractor(config_path: str) -> CombinedFeatureExtractor:
    """
    Load config from file and create CombinedFeatureExtractor
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured CombinedFeatureExtractor
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return create_combined_extractor(config)


if __name__ == "__main__":
    # Test the combined feature extractor
    import logging
    import tempfile
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test config
    test_config = {
        'ssl_model': {
            'name': 'microsoft/wavlm-large',
            'layer_index': 14,
            'freeze': True,
            'cache_dir': './cache/models/ssl'
        },
        'speaker_model': {
            'source': 'speechbrain/spkrec-ecapa-voxceleb',
            'freeze': True,
            'cache_dir': './cache/models/speaker'
        },
        'feature_extraction': {
            'num_frames': 100,
            'frame_sampling': 'random',
            'pca_components': 128,
            'fit_pca_on_first_batch': True
        }
    }
    
    try:
        # Initialize combined extractor
        extractor = create_combined_extractor(test_config)
        
        print(f"Model info: {extractor.get_model_info()}")
        
        # Test with dummy audio
        batch_size = 2
        audio_length = 32000  # 2 seconds at 16kHz
        dummy_audio = torch.randn(batch_size, audio_length)
        
        # Extract features (with PCA fitting)
        print("Extracting features with PCA fitting...")
        features = extractor(dummy_audio, fit_pca=True)
        
        print(f"Input audio shape: {dummy_audio.shape}")
        print(f"SSL features shape: {features['ssl_features'].shape}")
        print(f"Speaker embeddings shape: {features['speaker_embeddings'].shape}")
        
        # Extract features again (PCA already fitted)
        print("Extracting features with fitted PCA...")
        features2 = extractor(dummy_audio)
        print(f"Speaker embeddings shape (fitted PCA): {features2['speaker_embeddings'].shape}")
        
        # Test PCA save/load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pca_path = f.name
        
        extractor.save_pca(pca_path)
        
        # Create new extractor and load PCA
        extractor2 = create_combined_extractor(test_config)
        extractor2.load_pca(pca_path)
        
        features3 = extractor2(dummy_audio)
        print(f"Features shape with loaded PCA: {features3['speaker_embeddings'].shape}")
        
        # Check dimensions match paper requirements
        dims = extractor.get_feature_dims()
        print(f"Feature dimensions: {dims}")
        print(f"Expected matrix shapes for training:")
        print(f"  S (SSL features): ({batch_size * dims['num_frames']}, {dims['ssl_dim']})")
        print(f"  D (Speaker embeddings): ({batch_size * dims['num_frames']}, {dims['speaker_dim']})")
        
        print("Combined feature extraction test completed successfully!")
        
        # Cleanup
        import os
        os.unlink(pca_path)
        
    except Exception as e:
        print(f"Test failed (expected if models not available): {e}")
        import traceback
        traceback.print_exc()
