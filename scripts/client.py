#!/usr/bin/env python3
"""
Example client for Eta-WavLM Decomposition API

Demonstrates how to send WAV files to the API and receive decomposition results.
"""

import requests
import base64
import json
import numpy as np
from pathlib import Path
import argparse
import wave


def validate_wav_file(wav_path: str) -> dict:
    """Validate and get info about WAV file"""
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            info = {
                'sample_rate': wav_file.getframerate(),
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'frames': wav_file.getnframes(),
                'duration': wav_file.getnframes() / wav_file.getframerate()
            }
        return info
    except Exception as e:
        raise ValueError(f"Invalid WAV file: {e}")


def wav_to_base64(wav_path: str) -> str:
    """Convert WAV file to base64 string"""
    # First validate it's a proper WAV file
    wav_info = validate_wav_file(wav_path)
    print(f"WAV file info: {wav_info['sample_rate']}Hz, {wav_info['channels']}ch, "
          f"{wav_info['duration']:.2f}s")
    
    # Read raw WAV bytes (including header)
    with open(wav_path, 'rb') as wav_file:
        wav_bytes = wav_file.read()
        wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
    return wav_b64


def reconstruct_tensor(tensor_dict: dict) -> np.ndarray:
    """Reconstruct numpy array from API response"""
    shape = tensor_dict["shape"]
    data = np.array(tensor_dict["data"], dtype=tensor_dict["dtype"])
    return data.reshape(shape)


def call_eta_wavlm_api(wav_path: str, api_url: str = "http://localhost:8000/predict") -> dict:
    """
    Send WAV file to Eta-WavLM API and get decomposition
    
    Args:
        wav_path: Path to WAV file
        api_url: API endpoint URL
    
    Returns:
        Decomposition results with SSL, speaker, and eta features
    """
    
    # Validate WAV file format
    if not wav_path.lower().endswith('.wav'):
        print("Warning: File doesn't have .wav extension. Proceeding anyway...")
    
    # Convert WAV to base64
    print(f"Loading WAV file from {wav_path}...")
    try:
        wav_b64 = wav_to_base64(wav_path)
    except Exception as e:
        print(f"Error loading WAV file: {e}")
        return None
    
    # Prepare request
    request_data = {
        "wav_b64": wav_b64
    }
    
    # Send request
    print("Sending request to API...")
    try:
        response = requests.post(api_url, json=request_data, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        return None
    
    if response.status_code == 200:
        result = response.json()
        print("Decomposition successful!")
        return result
    else:
        print(f"API Error: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"Error details: {error_detail}")
        except:
            print(response.text)
        return None


def analyze_decomposition(result: dict):
    """Analyze and print decomposition results"""
    
    if not result:
        return
    
    print("\n" + "="*70)
    print("DECOMPOSITION ANALYSIS")
    print("="*70)
    
    # Extract components
    ssl_features = reconstruct_tensor(result["ssl_features"])
    speaker_embedding = reconstruct_tensor(result["speaker_embedding"])
    eta_features = reconstruct_tensor(result["eta_features"])
    wav_info = result["wav_info"]
    metadata = result["metadata"]
    
    # Print WAV file information
    print("WAV File Information:")
    print(f"  Original: {wav_info['original_sample_rate']}Hz, "
          f"{wav_info['original_channels']}ch, "
          f"{wav_info['original_duration_seconds']:.2f}s, "
          f"{wav_info['original_bits_per_sample']}bit")
    print(f"  File size: {wav_info['original_file_size']} bytes")
    print(f"  Audio format: {wav_info['audio_format']} "
          f"({'PCM' if wav_info['audio_format'] == 1 else 'Compressed'})")
    
    processing = wav_info['processing_applied']
    if processing['resampled'] or processing['converted_to_mono']:
        print("  Processing applied:")
        if processing['resampled']:
            print(f"    - Resampled: {processing['original_rate']}Hz → {processing['target_rate']}Hz")
        if processing['converted_to_mono']:
            print(f"    - Converted to mono")
    else:
        print("  No processing required")
    print()
    
    # Print feature information
    print(f"Processed Audio: {wav_info['processed_sample_rate']}Hz, "
          f"{wav_info['processed_samples']} samples")
    print()
    
    print(f"SSL Features (WavLM):")
    print(f"  Shape: {ssl_features.shape}")
    print(f"  Mean: {ssl_features.mean():.6f}")
    print(f"  Std:  {ssl_features.std():.6f}")
    print(f"  Range: [{ssl_features.min():.6f}, {ssl_features.max():.6f}]")
    print()
    
    print(f"Speaker Embedding (ECAPA-TDNN):")
    print(f"  Shape: {speaker_embedding.shape}")
    print(f"  Mean: {speaker_embedding.mean():.6f}")
    print(f"  Std:  {speaker_embedding.std():.6f}")
    print(f"  Range: [{speaker_embedding.min():.6f}, {speaker_embedding.max():.6f}]")
    print()
    
    print(f"Eta Features (Speaker-Independent Content):")
    print(f"  Shape: {eta_features.shape}")
    print(f"  Mean: {eta_features.mean():.6f}")
    print(f"  Std:  {eta_features.std():.6f}")
    print(f"  Range: [{eta_features.min():.6f}, {eta_features.max():.6f}]")
    print()
    
    # Calculate norms and ratios
    ssl_norm = np.linalg.norm(ssl_features)
    speaker_norm = np.linalg.norm(speaker_embedding)
    eta_norm = np.linalg.norm(eta_features)
    
    print("Feature Magnitudes:")
    print(f"  SSL features norm: {ssl_norm:.6f}")
    print(f"  Speaker embedding norm: {speaker_norm:.6f}")
    print(f"  Eta features norm: {eta_norm:.6f}")
    print(f"  Speaker/SSL ratio: {speaker_norm/ssl_norm:.6f}")
    print(f"  Eta/SSL ratio: {eta_norm/ssl_norm:.6f}")
    print()
    
    # Calculate decomposition quality metrics
    # Note: This assumes eta + f(speaker) ≈ ssl features
    print("Decomposition Quality:")
    correlation = np.corrcoef(ssl_features.flatten(), eta_features.flatten())[0,1]
    print(f"  SSL-Eta correlation: {correlation:.6f}")
    
    # Check if decomposition maintains content
    print(f"  Model version: {metadata['model_version']}")
    print(f"  SSL model: {metadata['ssl_model']}")
    print(f"  Speaker encoder: {metadata['speaker_encoder']}")


def main():
    parser = argparse.ArgumentParser(description="Test Eta-WavLM Decomposition API with WAV files")
    parser.add_argument("wav_path", help="Path to WAV file")
    parser.add_argument("--api-url", default="http://localhost:8000/predict", 
                       help="API endpoint URL")
    parser.add_argument("--save-results", help="Save results to JSON file")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate WAV file, don't call API")
    
    args = parser.parse_args()
    
    # Check WAV file exists
    if not Path(args.wav_path).exists():
        print(f"Error: WAV file not found: {args.wav_path}")
        return
    
    # Validate WAV file
    try:
        validate_wav_file(args.wav_path)
        print(f"WAV file validation passed: {args.wav_path}")
    except Exception as e:
        print(f"WAV file validation failed: {e}")
        return
    
    if args.validate_only:
        print("Validation complete. Exiting.")
        return
    
    # Call API
    result = call_eta_wavlm_api(args.wav_path, args.api_url)
    
    if result:
        # Analyze results
        analyze_decomposition(result)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                # Don't save the raw tensor data, just metadata for file size reasons
                summary_result = {
                    'wav_info': result['wav_info'],
                    'metadata': result['metadata'],
                    'feature_shapes': {
                        'ssl_features': result['ssl_features']['shape'],
                        'speaker_embedding': result['speaker_embedding']['shape'],
                        'eta_features': result['eta_features']['shape']
                    }
                }
                json.dump(summary_result, f, indent=2)
            print(f"\nResults summary saved to: {args.save_results}")
            print("(Note: Full tensor data not saved due to size. Use API response directly if needed.)")


if __name__ == "__main__":
    main()
