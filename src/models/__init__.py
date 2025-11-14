"""
Model components for Eta-WavLM

This package contains the core EtaWavLM model implementation and PyTorch Lightning
wrapper for training the linear decomposition: s = f(d) + η
"""

# Core EtaWavLM Model
from .eta_wavlm_model import (
    EtaWavLMModel,
    create_eta_wavlm_model
)

# PyTorch Lightning Module
from .eta_wavlm_lightning import (
    EtaWavLMLightningModule, 
    create_eta_wavlm_lightning_module
)

__all__ = [
    # Core Model
    "EtaWavLMModel",
    "create_eta_wavlm_model",
    
    # Lightning Module
    "EtaWavLMLightningModule",
    "create_eta_wavlm_lightning_module"
]
