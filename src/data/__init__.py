"""
Data handling components for Eta-WavLM

This package contains dataset classes and Lightning DataModules for handling
audio data for Eta-WavLM training and evaluation.
"""

from .audio_dataset import AudioDataset, collate_fn
from .data_module import EtaWavLMDataModule, create_datamodule_from_config

__all__ = [
    "AudioDataset",
    "collate_fn", 
    "EtaWavLMDataModule",
    "create_datamodule_from_config"
]
