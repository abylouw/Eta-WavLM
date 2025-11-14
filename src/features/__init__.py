"""
Feature extraction components for Eta-WavLM

This package contains SSL feature extractors and speaker encoders for extracting
the 's' and 'd' components used in the Eta-WavLM linear decomposition: s = f(d) + η
"""

# SSL Feature Extractors
from .ssl_extractor import (
    BaseSSLExtractor,
    WavLMExtractor,
    Wav2Vec2Extractor,
    HuBERTExtractor,
    FrameSampler,
    create_ssl_extractor,
    create_frame_sampler
)

# Speaker Encoders
from .speaker_encoder import (
    BaseSpeakerEncoder,
    ECAPATDNNEncoder,
    ResemblyzerEncoder,
    WavLMSVEncoder,
    SpeakerEmbeddingReplicator,
    create_speaker_encoder,
    create_speaker_replicator,
    verify_speaker_encoder
)

# Combined Feature Extractor
from .combined_extractor import (
    CombinedFeatureExtractor,
    create_combined_extractor,
    load_config_and_create_extractor
)

__all__ = [
    # SSL Extractors
    "BaseSSLExtractor",
    "WavLMExtractor", 
    "Wav2Vec2Extractor",
    "HuBERTExtractor",
    "FrameSampler",
    "create_ssl_extractor",
    "create_frame_sampler",
    
    # Speaker Encoders
    "BaseSpeakerEncoder",
    "ECAPATDNNEncoder",
    "ResemblyzerEncoder", 
    "WavLMSVEncoder",
    "SpeakerEmbeddingReplicator",
    "create_speaker_encoder",
    "create_speaker_replicator",
    "verify_speaker_encoder",
    
    # Combined Extractor
    "CombinedFeatureExtractor",
    "create_combined_extractor", 
    "load_config_and_create_extractor"
]
