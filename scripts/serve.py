#!/usr/bin/env python3
"""
Eta-WavLM Decomposition API using LitServe

Provides a web service for decomposing audio into SSL features, 
speaker embeddings, and eta (speaker-independent) representations.
"""

import litserve as ls
import torch
import torchaudio
import numpy as np
import base64
import io
import wave
import struct
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import yaml
import json
import sys
import soundfile as sf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_wav_header(wav_bytes: bytes) -> Dict[str, Any]:
    """
    Parse WAV file header to extract metadata
    
    Args:
        wav_bytes: Raw WAV file bytes
        
    Returns:
        Dictionary with WAV file metadata
    """
    if len(wav_bytes) < 44:
        raise ValueError("Invalid WAV file: too short")
    
    # Check RIFF header
    if wav_bytes[:4] != b'RIFF':
        raise ValueError("Invalid WAV file: missing RIFF header")
    
    if wav_bytes[8:12] != b'WAVE':
        raise ValueError("Invalid WAV file: not a WAVE file")
    
    # Find fmt chunk
    offset = 12
    fmt_chunk_found = False
    
    while offset < len(wav_bytes) - 8:
        chunk_id = wav_bytes[offset:offset+4]
        chunk_size = struct.unpack('<I', wav_bytes[offset+4:offset+8])[0]
        
        if chunk_id == b'fmt ':
            fmt_chunk_found = True
            fmt_data = wav_bytes[offset+8:offset+8+chunk_size]
            
            if len(fmt_data) < 16:
                raise ValueError("Invalid WAV file: fmt chunk too short")
            
            # Parse fmt chunk
            audio_format = struct.unpack('<H', fmt_data[0:2])[0]
            num_channels = struct.unpack('<H', fmt_data[2:4])[0]
            sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
            byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
            block_align = struct.unpack('<H', fmt_data[12:14])[0]
            bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            
            break
        
        offset += 8 + chunk_size
        # Ensure word alignment
        if chunk_size % 2 == 1:
            offset += 1
    
    if not fmt_chunk_found:
        raise ValueError("Invalid WAV file: no fmt chunk found")
    
    # Find data chunk to get duration
    offset = 12
    data_size = 0
    
    while offset < len(wav_bytes) - 8:
        chunk_id = wav_bytes[offset:offset+4]
        chunk_size = struct.unpack('<I', wav_bytes[offset+4:offset+8])[0]
        
        if chunk_id == b'data':
            data_size = chunk_size
            break
        
        offset += 8 + chunk_size
        if chunk_size % 2 == 1:
            offset += 1
    
    # Calculate duration
    bytes_per_sample = bits_per_sample // 8
    total_samples = data_size // (num_channels * bytes_per_sample)
    duration_seconds = total_samples / sample_rate
    
    return {
        'audio_format': audio_format,
        'num_channels': num_channels,
        'sample_rate': sample_rate,
        'byte_rate': byte_rate,
        'block_align': block_align,
        'bits_per_sample': bits_per_sample,
        'data_size': data_size,
        'total_samples': total_samples,
        'duration_seconds': duration_seconds,
        'file_size': len(wav_bytes)
    }


class EtaWavLMDecompositionAPI(ls.LitAPI):
    """
    LitServe API for Eta-WavLM audio decomposition.
    
    Takes base64-encoded audio and returns:
    - SSL (WavLM) representations
    - Speaker embeddings 
    - Eta (speaker-independent) representations
    """

    def __init__(self, model_path=None, config_path=None):
        super().__init__()
        self.model_path = model_path
        self.config_path = config_path

        
    def setup(self, device: str):
        """Initialize the model and move to device"""
        self.device = device
        
        # Import your project modules
        try:
            from src.models import create_eta_wavlm_lightning_module
            from src.utils.misc import load_eval_config, get_project_root
            import pickle  
        except ImportError as e:
            logger.error(f"Failed to import project modules: {e}")
            raise

        if self.model_path:
            model_path = Path(self.model_path)
        else:
            model_path = get_project_root() / "outputs"/ "eta_wavlm_full_training" / "eta_wavlm_full_training_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.config_path:
            config_path = Path(self.config_path)
        else:
            config_path = get_project_root() / "configs" / "model" / "eta_wavlm.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        self.config = load_eval_config(config_path)
        self.model = create_eta_wavlm_lightning_module(self.config)

        logger.info(f"Loading model from {model_path}")
        self.model.load_model(model_path)

        # Explicitly load PCA file
        project_root = get_project_root()
    
        # Try multiple possible PCA locations
        pca_candidates = [
            project_root / "pca_model.pkl",  # Default name from your config
            model_path.parent / "pca_model.pkl",  # Same directory as model
            project_root / "outputs" / "eta_wavlm_full_training" / "pca_model.pkl",  # In model output dir
            project_root / "speaker_pca_model.pkl",  # Alternative name
        ]

        pca_loaded = False
        for pca_path in pca_candidates:
            if pca_path.exists():
                try:
                    logger.info(f"Trying to load PCA from: {pca_path}")
                    with open(pca_path, 'rb') as f:
                        self.model.model.pca_model = pickle.load(f)
                    self.model.model.pca_fitted = True
                    logger.info(f"Successfully loaded PCA from: {pca_path}")
                    pca_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load PCA from {pca_path}: {e}")
                    continue

        if not pca_loaded:
            logger.error("Could not find PCA file. Searched:")
            for path in pca_candidates:
                logger.error(f"  {path}")
            raise FileNotFoundError("PCA model file not found")
        
        self.model.to(device)
        self.model.eval()
        
        # Verify model is ready for inference
        if not self.model.model.parameters_solved:
            raise RuntimeError("Model parameters not solved")
        if not self.model.model.pca_fitted:
            raise RuntimeError("PCA model not loaded")
                
        logger.info("Eta-WavLM model loaded successfully")
    
    def decode_request(self, request: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode base64 WAV file to tensor and extract metadata
        
        Expected request format:
        {
            "wav_b64": "base64-encoded-wav-file"
        }
        
        Returns:
            Tuple of (waveform_tensor, wav_metadata)
        """
        try:
            # Extract WAV data
            if "wav_b64" not in request:
                raise ValueError("Missing 'wav_b64' in request")
            
            wav_b64 = request["wav_b64"]
            
            # Remove data URL prefix if present
            if wav_b64.startswith('data:audio'):
                wav_b64 = wav_b64.split(',', 1)[1]
            
            # Decode base64 WAV file
            wav_bytes = base64.b64decode(wav_b64)
            
            # Parse WAV header to extract metadata
            wav_metadata = parse_wav_header(wav_bytes)
            
            logger.info(f"WAV file info: {wav_metadata['sample_rate']}Hz, "
                       f"{wav_metadata['num_channels']}ch, "
                       f"{wav_metadata['duration_seconds']:.2f}s, "
                       f"{wav_metadata['bits_per_sample']}bit")
            
            # Load audio from bytes using torchaudio
            wav_buffer = io.BytesIO(wav_bytes)
            data, sample_rate = sf.read(wav_buffer)

            # Verify sample rate matches header
            if sample_rate != wav_metadata['sample_rate']:
                logger.warning(f"Sample rate mismatch: header={wav_metadata['sample_rate']}, "
                             f"detected={sample_rate}. Using header value.")

            waveform = torch.from_numpy(data)

            # Check if the waveform is a 1D tensor (mono file)
            if waveform.ndim == 1:
                # Add a channel dimension: (Length,) -> (1, Length)
                waveform = waveform.unsqueeze(0)
            # Check if the waveform is a 2D tensor (stereo/multi-channel file)
            elif waveform.ndim == 2:
                # soundfile returns (Length, Channels). Transpose to (Channels, Length)
                # The current shape check is also wrong for 2D data (see point 2)
                waveform = waveform.T
    
                # Now, if we need to convert to mono, it's done on the new shape (C, L)
                if waveform.shape[0] > 1:
                    logger.info(f"Converting from {waveform.shape[0]} channels to mono")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.to(torch.float32) 
                       
            # Resample to 16kHz if needed (required for WavLM)
            target_sr = 16000
            if wav_metadata['sample_rate'] != target_sr:
                logger.info(f"Resampling from {wav_metadata['sample_rate']}Hz to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(wav_metadata['sample_rate'], target_sr)
                waveform = resampler(waveform)
                # Update metadata for processed audio
                wav_metadata['processed_sample_rate'] = target_sr
                wav_metadata['processed_samples'] = waveform.shape[-1]
            else:
                wav_metadata['processed_sample_rate'] = wav_metadata['sample_rate']
                wav_metadata['processed_samples'] = waveform.shape[-1]
            
            # Add batch dimension and move to device
            waveform = waveform.unsqueeze(0).to(self.device)
            
            return waveform, wav_metadata
            
        except Exception as e:
            logger.error(f"Failed to decode WAV file: {e}")
            raise ValueError(f"WAV decoding error: {str(e)}")
    
    def predict(self, inputs: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform decomposition on the audio waveform
        
        Args:
            inputs: Tuple of (waveform, wav_metadata)
        
        Returns:
            Dictionary with decomposition results and metadata
        """
        try:
            waveform, wav_metadata = inputs
            
            with torch.no_grad():
                # Extract components using your model
                decomposition = self.model.model.generate_eta_representations(
                    waveform,
                    return_components=True
                )
                
                return {
                    "ssl_features": decomposition["ssl_features"],
                    "speaker_embedding": decomposition["speaker_embeddings"], 
                    "eta_features": decomposition["eta"],
                    "wav_metadata": wav_metadata,
                    "audio_length": torch.tensor(waveform.shape[-1])
                }
                
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            raise RuntimeError(f"Model inference error: {str(e)}")
    
    def encode_response(self, decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tensors to JSON-serializable format
        
        Returns:
        {
            "ssl_features": {...},
            "speaker_embedding": {...},
            "eta_features": {...},
            "wav_info": {...},
            "metadata": {...}
        }
        """
        try:
            def tensor_to_dict(tensor: torch.Tensor) -> Dict[str, Any]:
                """Convert tensor to dictionary with shape and data"""
                numpy_array = tensor.cpu().numpy()
                return {
                    "shape": list(numpy_array.shape),
                    "dtype": str(numpy_array.dtype),
                    "data": numpy_array.flatten().tolist()  # Flatten for JSON
                }
            
            wav_metadata = decomposition["wav_metadata"]
            
            response = {
                "ssl_features": tensor_to_dict(decomposition["ssl_features"]),
                "speaker_embedding": tensor_to_dict(decomposition["speaker_embedding"]),
                "eta_features": tensor_to_dict(decomposition["eta_features"]),
                "wav_info": {
                    # Original WAV file info
                    "original_sample_rate": wav_metadata["sample_rate"],
                    "original_channels": wav_metadata["num_channels"],
                    "original_bits_per_sample": wav_metadata["bits_per_sample"],
                    "original_duration_seconds": wav_metadata["duration_seconds"],
                    "original_file_size": wav_metadata["file_size"],
                    "audio_format": wav_metadata["audio_format"],
                    
                    # Processed audio info
                    "processed_sample_rate": wav_metadata["processed_sample_rate"],
                    "processed_samples": wav_metadata["processed_samples"],
                    "processing_applied": {
                        "resampled": wav_metadata["sample_rate"] != wav_metadata["processed_sample_rate"],
                        "converted_to_mono": wav_metadata["num_channels"] > 1,
                        "original_rate": wav_metadata["sample_rate"],
                        "target_rate": wav_metadata["processed_sample_rate"]
                    }
                },
                "metadata": {
                    "audio_length_samples": int(decomposition["audio_length"].item()),
                    "model_version": "eta-wavlm-v1",
                    "ssl_model": "WavLM-Large",
                    "speaker_encoder": "ECAPA-TDNN",
                    "processing_timestamp": None  # Could add timestamp if needed
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Response encoding failed: {e}")
            raise RuntimeError(f"Response encoding error: {str(e)}")


def create_app(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000
) -> ls.LitServer:
    """
    Create and configure the LitServe application
    
    Args:
        model_path: Path to model checkpoint (optional, uses default)
        config_path: Path to config file (optional, uses default)
        host: Server host
        port: Server port
    
    Returns:
        Configured LitServe application
    """
    
    # Create API instance
    api = EtaWavLMDecompositionAPI(model_path=model_path, config_path=config_path)
    
    # Create LitServe app
    app = ls.LitServer(api)
    
    return app


def main():
    """Run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Eta-WavLM Decomposition API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--config-path", help="Path to config file")
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(
        model_path=args.model_path,
        config_path=args.config_path
    )
    
    # Run server
    logger.info(f"Starting Eta-WavLM API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
