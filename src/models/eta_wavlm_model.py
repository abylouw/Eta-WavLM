"""
Unified EtaWavLM Model Implementation

Implements the paper's two-stage approach with online accumulation:
Stage 1: Extract speaker embeddings, fit PCA, save transformed embeddings  
Stage 2: Online SSL extraction with matrix accumulation, solve linear system
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import logging
from pathlib import Path
import time
import pickle
import hashlib

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

from ..features.ssl_extractor import create_ssl_extractor, create_frame_sampler
from ..features.speaker_encoder import create_speaker_encoder

logger = logging.getLogger(__name__)


class EtaWavLMModel(nn.Module):
    """
    Unified EtaWavLM Model with Two-Stage Training
    
    Stage 1: Speaker embeddings extraction, PCA fitting, and storage
    Stage 2: Online SSL extraction with matrix accumulation and solving
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Configuration
        decomp_config = config.get('decomposition', {})
        feature_config = config.get('feature_extraction', {})
        training_config = config.get('training', {})
        
        self.regularization = float(decomp_config.get('regularization', 1e-6))
        self.solver = decomp_config.get('solver', 'cholesky')
        self.num_frames = feature_config.get('num_frames', 100)
        self.pca_components = feature_config.get('pca_components', 128)
        
        # File paths for saving/loading intermediate results
        self.embeddings_save_path = training_config.get('embeddings_save_path', 'speaker_embeddings.pkl')
        self.pca_save_path = training_config.get('pca_save_path', 'pca_model.pkl')
        
        # Initialize extractors
        self.ssl_extractor = create_ssl_extractor(config)
        self.speaker_encoder = create_speaker_encoder(config)
        self.frame_sampler = create_frame_sampler(config)
        
        # Fixed dimensions
        ssl_output_dim = self.ssl_extractor.get_output_dim()
        logger.info(f"SSL extractor output dim: {ssl_output_dim}")
        
        # Ensure dimensions are always integers
        self.ssl_dim = int(ssl_output_dim)  # Q
        self.speaker_dim = int(self.pca_components)  # P (after PCA)
        self.speaker_dim_with_bias = int(self.speaker_dim + 1)  # P + 1
        
        logger.info(f"Model dimensions - SSL: {self.ssl_dim}, Speaker: {self.speaker_dim}")
        
        # Training state
        self.stage = None  # 'stage1', 'stage2', 'solved'
        self.pca_fitted = False
        self.parameters_solved = False
        
        # Stage 1: Speaker embedding collection
        self.speaker_embeddings_storage = {}  # utterance_id -> pca_transformed_embedding
        self.pca_model = None
        
        # Stage 2: Online accumulation
        self.G_accumulator = None  # (P+1, P+1) - accumulated D̃ᵀD̃
        self.H_accumulator = None  # (P+1, Q) - accumulated D̃ᵀS
        self.n_accumulated_samples = 0
        self.n_accumulated_utterances = 0
        
        # Final linear system
        self.A_star = None  # (P, Q)
        self.b_star = None  # (Q,)
        self.A_tilde_star = None  # (P+1, Q)
        
        logger.info(f"Initialized EtaWavLM Model:")
        logger.info(f"  SSL dim (Q): {self.ssl_dim}")
        logger.info(f"  Speaker dim (P): {self.speaker_dim}")
        logger.info(f"  Num frames (L): {self.num_frames}")
    
    def _generate_utterance_id(self, waveforms: torch.Tensor, batch_idx: int) -> List[str]:
        """Generate unique utterance IDs for tracking across stages"""
        batch_size = waveforms.shape[0]
        ids = []
        
        for i in range(batch_size):
            # Create hash from waveform + batch info for unique ID
            waveform_bytes = waveforms[i].cpu().numpy().tobytes()
            id_string = f"{batch_idx}_{i}_{hashlib.md5(waveform_bytes).hexdigest()[:8]}"
            ids.append(id_string)
        
        return ids
    
    # ========== STAGE 1: SPEAKER EMBEDDINGS & PCA ==========
    
    def start_stage1(self):
        """Start Stage 1: Speaker embedding collection"""
        logger.info("Starting Stage 1: Collecting speaker embeddings for PCA...")
        
        self.stage = 'stage1'
        self.speaker_embeddings_storage.clear()
        self.pca_fitted = False
        self.parameters_solved = False
        
        logger.info("Stage 1 ready for speaker embedding collection")
    
    def collect_speaker_embeddings_stage1(
        self, 
        waveforms: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """Stage 1: Collect speaker embeddings from utterances"""
        if self.stage != 'stage1':
            raise RuntimeError("Must call start_stage1() first")
        
        # Generate utterance IDs
        utterance_ids = self._generate_utterance_id(waveforms, batch_idx)
        
        # Extract speaker embeddings (one per utterance)
        speaker_embeddings = self.speaker_encoder(waveforms, lengths)  # (batch_size, embedding_dim)
        
        # Store raw embeddings with IDs (will be PCA-transformed later)
        for i, utterance_id in enumerate(utterance_ids):
            embedding = speaker_embeddings[i].cpu().numpy()
            self.speaker_embeddings_storage[utterance_id] = embedding
        
        return {
            'batch_size': len(utterance_ids),
            'total_utterances_collected': len(self.speaker_embeddings_storage),
            'utterance_ids': utterance_ids
        }
    
    def fit_pca_and_save_stage1(self) -> Dict[str, Any]:
        """Stage 1: Fit PCA on collected embeddings and save everything"""
        if self.stage != 'stage1' or not self.speaker_embeddings_storage:
            raise RuntimeError("Must collect speaker embeddings first")
        
        logger.info("Fitting PCA on collected speaker embeddings...")
        
        # Collect all embeddings for PCA fitting
        all_embeddings = []
        utterance_ids = list(self.speaker_embeddings_storage.keys())
        
        for utterance_id in utterance_ids:
            all_embeddings.append(self.speaker_embeddings_storage[utterance_id])
        
        embeddings_matrix = np.stack(all_embeddings)  # (num_utterances, embedding_dim)
        
        logger.info(f"Fitting PCA on {embeddings_matrix.shape[0]} utterances, "
                   f"dims {embeddings_matrix.shape[1]} → {self.pca_components}")
        
        # Fit PCA
        self.pca_model = PCA(n_components=self.pca_components, random_state=42)
        self.pca_model.fit(embeddings_matrix)
        self.pca_fitted = True
        
        # Apply PCA transformation to all embeddings
        transformed_embeddings = self.pca_model.transform(embeddings_matrix)
        
        # Update storage with transformed embeddings
        for i, utterance_id in enumerate(utterance_ids):
            self.speaker_embeddings_storage[utterance_id] = transformed_embeddings[i]
        
        # Save PCA model
        with open(self.pca_save_path, 'wb') as f:
            pickle.dump(self.pca_model, f)
        
        # Save transformed embeddings
        with open(self.embeddings_save_path, 'wb') as f:
            pickle.dump(self.speaker_embeddings_storage, f)
        
        # Log PCA info
        explained_var = np.sum(self.pca_model.explained_variance_ratio_)
        logger.info(f"PCA fitted: {explained_var:.3f} variance explained with {self.pca_components} components")
        logger.info(f"Stage 1 complete: saved PCA to {self.pca_save_path}, embeddings to {self.embeddings_save_path}")
        
        return {
            'num_utterances': len(self.speaker_embeddings_storage),
            'pca_components': self.pca_components,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'total_explained_variance': explained_var,
            'pca_path': self.pca_save_path,
            'embeddings_path': self.embeddings_save_path
        }
    
    # ========== STAGE 2: ONLINE SSL EXTRACTION & ACCUMULATION ==========
    
    def start_stage2(self, embeddings_path: str):
        """Start Stage 2: Load embeddings and initialize accumulators"""
        logger.info("Starting Stage 2: SSL extraction with online accumulation...")
        
        # Load speaker embeddings
        if not Path(embeddings_path).exists():
            raise RuntimeError(f"Speaker embeddings not found: {embeddings_path}")
        
        with open(embeddings_path, 'rb') as f:
            self.speaker_embeddings_storage = pickle.load(f)
        
        # Load PCA model  
        if not Path(self.pca_save_path).exists():
            raise RuntimeError(f"PCA model not found: {self.pca_save_path}")
        
        with open(self.pca_save_path, 'rb') as f:
            self.pca_model = pickle.load(f)
        
        self.pca_fitted = True
        self.stage = 'stage2'
        
        # Initialize matrix accumulators for online computation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G_accumulator = torch.zeros(
            self.speaker_dim_with_bias, self.speaker_dim_with_bias, 
            dtype=torch.float64, device=device
        )
        self.H_accumulator = torch.zeros(
            self.speaker_dim_with_bias, self.ssl_dim, 
            dtype=torch.float64, device=device
        )
        self.n_accumulated_samples = 0
        self.n_accumulated_utterances = 0
        
        logger.info(f"Stage 2 ready:")
        logger.info(f"  Speaker embeddings: {len(self.speaker_embeddings_storage)}")
        logger.info(f"  Accumulators: G {self.G_accumulator.shape}, H {self.H_accumulator.shape}")
    
    def collect_ssl_features_stage2(
        self, 
        waveforms: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """Stage 2: Extract SSL features and update accumulators online"""
        if self.stage != 'stage2':
            raise RuntimeError("Must call start_stage2() first")
        
        # Generate utterance IDs
        utterance_ids = self._generate_utterance_id(waveforms, batch_idx)
                
        # Extract SSL features
        ssl_features = self.ssl_extractor(waveforms, lengths)
        ssl_features_sampled = self.frame_sampler.sample_frames(ssl_features)
                
        # Get speaker embeddings for these utterances
        batch_speaker_embeddings = []
        valid_utterances = 0
        
        for utterance_id in utterance_ids:
            if utterance_id in self.speaker_embeddings_storage:
                embedding = self.speaker_embeddings_storage[utterance_id]
                batch_speaker_embeddings.append(embedding)
                valid_utterances += 1
            else:
                # Generate new speaker embedding if not found
                logger.warning(f"Speaker embedding not found for {utterance_id}, extracting...")
                idx = len(batch_speaker_embeddings)
                if idx < len(waveforms):
                    speaker_emb = self.speaker_encoder(waveforms[idx:idx+1], 
                                                       lengths[idx:idx+1] if lengths is not None else None)
                    speaker_emb_np = speaker_emb[0].cpu().numpy()
                    speaker_emb_pca = self.pca_model.transform(speaker_emb_np.reshape(1, -1))[0]
                    batch_speaker_embeddings.append(speaker_emb_pca)
                    valid_utterances += 1
        
        if valid_utterances == 0:
            return {
                'batch_size': 0,
                'total_ssl_collected': self.n_accumulated_samples,
                'utterances_processed': self.n_accumulated_utterances
            }
        
        # Prepare data for accumulation
        batch_ssl_features = ssl_features_sampled[:valid_utterances]  # (batch_size, num_frames, ssl_dim)
        batch_speaker_embeddings = np.stack(batch_speaker_embeddings)  # (batch_size, speaker_dim)
        
        # Flatten and replicate speaker embeddings across frames
        batch_size, num_frames, ssl_dim = batch_ssl_features.shape
        
        # Flatten SSL features: (batch_size * num_frames, ssl_dim)
        S_batch = batch_ssl_features.reshape(-1, ssl_dim)
                
        # Replicate speaker embeddings across frames: (batch_size * num_frames, speaker_dim)
        speaker_embeddings_rep = np.repeat(batch_speaker_embeddings, num_frames, axis=0)
                
        # Convert to tensors
        device = self.G_accumulator.device
        S_tensor = S_batch.to(device).to(torch.float64)
        D_tensor = torch.from_numpy(speaker_embeddings_rep).to(device).to(torch.float64)
        
        # Add bias column to create D̃
        bias_column = torch.ones(D_tensor.shape[0], 1, device=device, dtype=torch.float64)
        D_tilde = torch.cat([D_tensor, bias_column], dim=1)
        
        # Update accumulators: G += D̃ᵀD̃, H += D̃ᵀS
        self.G_accumulator += torch.mm(D_tilde.t(), D_tilde)
        self.H_accumulator += torch.mm(D_tilde.t(), S_tensor)
        
        # Update counters
        self.n_accumulated_samples += S_tensor.shape[0]
        self.n_accumulated_utterances += valid_utterances
        
        # Clean up
        del S_tensor, D_tensor, D_tilde
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'batch_size': valid_utterances,
            'total_ssl_collected': self.n_accumulated_samples,
            'utterances_processed': self.n_accumulated_utterances
        }
    
    def solve_decomposition_stage2(self) -> Dict[str, Any]:
        """Stage 2: Solve decomposition using accumulated matrices"""
        if self.stage != 'stage2':
            raise RuntimeError("Must be in stage 2")
        
        if self.n_accumulated_samples == 0:
            return {'success': False, 'error': 'No data accumulated'}
        
        logger.info(f"Solving decomposition with accumulated data...")
        logger.info(f"Accumulated samples: {self.n_accumulated_samples}")
        logger.info(f"Accumulated utterances: {self.n_accumulated_utterances}")
        logger.info(f"Matrix shapes: G{self.G_accumulator.shape}, H{self.H_accumulator.shape}")
        
        try:
            # Add regularization to G
            regularization_matrix = self.regularization * torch.eye(
                self.speaker_dim_with_bias,
                device=self.G_accumulator.device,
                dtype=self.G_accumulator.dtype
            )
            G_reg = self.G_accumulator + regularization_matrix
            
            # Solve G_reg * A = H
            if self.solver == 'cholesky':
                try:
                    L = torch.linalg.cholesky(G_reg)
                    y = torch.linalg.solve_triangular(L, self.H_accumulator, upper=False)
                    A_tilde_solution = torch.linalg.solve_triangular(L.t(), y, upper=True)
                except Exception as e:
                    logger.warning(f"Cholesky failed ({e}), falling back to lstsq")
                    A_tilde_solution = torch.linalg.lstsq(G_reg, self.H_accumulator).solution
            elif self.solver == 'lstsq':
                A_tilde_solution = torch.linalg.lstsq(G_reg, self.H_accumulator).solution
            elif self.solver == 'pseudo_inverse':
                G_pinv = torch.linalg.pinv(G_reg)
                A_tilde_solution = torch.mm(G_pinv, self.H_accumulator)
            else:
                A_tilde_solution = torch.linalg.solve(G_reg, self.H_accumulator)
            
            # Extract A* and b*
            A_star = A_tilde_solution[:-1, :]  # (P, Q)
            b_star = A_tilde_solution[-1, :]   # (Q,)
            
            # Convert to float32 and store
            self.A_star = A_star.to(torch.float32)
            self.b_star = b_star.to(torch.float32)
            self.A_tilde_star = A_tilde_solution.to(torch.float32)
            
            # Compute solution statistics
            condition_number = torch.linalg.cond(G_reg).item()
            residual_norm = torch.norm(torch.mm(G_reg, A_tilde_solution) - self.H_accumulator).item()
            
            self.parameters_solved = True
            self.stage = 'solved'
            
            logger.info("Decomposition solved successfully!")
            logger.info(f"  A* shape: {self.A_star.shape}")
            logger.info(f"  b* shape: {self.b_star.shape}")
            logger.info(f"  Condition number: {condition_number:.2e}")
            logger.info(f"  Residual norm: {residual_norm:.2e}")
            
            return {
                'success': True,
                'condition_number': condition_number,
                'residual_norm': residual_norm,
                'n_samples': self.n_accumulated_samples,
                'n_utterances': self.n_accumulated_utterances,
                'A_star_shape': list(self.A_star.shape),
                'b_star_shape': list(self.b_star.shape)
            }
            
        except Exception as e:
            logger.error(f"Failed to solve decomposition: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========== INFERENCE ==========
    
    def generate_eta_representations(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate eta representations using solved parameters"""
        if not self.parameters_solved:
            raise RuntimeError("Parameters not solved yet")
        
        # Extract SSL features
        ssl_features = self.ssl_extractor(waveforms, lengths)
        ssl_features_sampled = self.frame_sampler.sample_frames(ssl_features)
        
        # Extract speaker embeddings and apply PCA
        speaker_embeddings_raw = self.speaker_encoder(waveforms, lengths)
        speaker_embeddings_np = speaker_embeddings_raw.cpu().numpy()
        speaker_embeddings_pca = self.pca_model.transform(speaker_embeddings_np)
        speaker_embeddings = torch.from_numpy(speaker_embeddings_pca).to(ssl_features.device).float()
        
        # Replicate across frames
        batch_size, num_frames = ssl_features_sampled.shape[0], ssl_features_sampled.shape[1]
        speaker_embeddings_rep = speaker_embeddings.unsqueeze(1).expand(-1, num_frames, -1)
        
        # Flatten
        ssl_features_flat = ssl_features_sampled.reshape(-1, ssl_features_sampled.shape[-1])
        speaker_embeddings_flat = speaker_embeddings_rep.reshape(-1, speaker_embeddings_rep.shape[-1])
        
        # Compute speaker component: d^T A* + b*
        speaker_component = torch.mm(speaker_embeddings_flat, self.A_star) + self.b_star
        
        # Compute eta: η = s - (d^T A* + b*)
        eta_representations = ssl_features_flat - speaker_component
        
        if return_components:
            components = {
                'eta': eta_representations,
                'ssl_features': ssl_features_flat,
                'speaker_component': speaker_component,
                'speaker_embeddings': speaker_embeddings_flat
            }
            
            # Add expanded speaker IDs if provided
            if speaker_ids is not None:
                speaker_ids_expanded = speaker_ids.repeat_interleave(num_frames)
                components['speaker_ids_expanded'] = speaker_ids_expanded
            
            return components
        else:
            return eta_representations
    
    # ========== SAVE/LOAD ==========
    
    def save_parameters(self, path: str):
        """Save learned parameters"""
        save_dict = {
            'A_star': self.A_star,
            'b_star': self.b_star,
            'A_tilde_star': self.A_tilde_star,
            'ssl_dim': self.ssl_dim,
            'speaker_dim': self.speaker_dim,
            'parameters_solved': self.parameters_solved,
            'stage': self.stage,
            'config': self.config,
            'pca_save_path': self.pca_save_path,
            'embeddings_save_path': self.embeddings_save_path,
            'n_accumulated_samples': self.n_accumulated_samples
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved parameters to {path}")
    
    def load_parameters(self, path: str):
        """Load learned parameters"""
        save_dict = torch.load(path, map_location='cpu')
        
        self.A_star = save_dict['A_star']
        self.b_star = save_dict['b_star'] 
        self.A_tilde_star = save_dict['A_tilde_star']
        self.parameters_solved = save_dict.get('parameters_solved', True)
        self.stage = save_dict.get('stage', 'solved')
        self.n_accumulated_samples = save_dict.get('n_accumulated_samples', 0)
        
        # Load PCA if available
        pca_path = save_dict.get('pca_save_path', self.pca_save_path)
        if Path(pca_path).exists():
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            self.pca_fitted = True
        
        logger.info(f"Loaded parameters from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'ssl_dim': self.ssl_dim,
            'speaker_dim': self.speaker_dim,
            'stage': self.stage,
            'pca_fitted': self.pca_fitted,
            'parameters_solved': self.parameters_solved,
            'num_stored_speaker_embeddings': len(self.speaker_embeddings_storage) if hasattr(self, 'speaker_embeddings_storage') else 0,
            'n_accumulated_samples': self.n_accumulated_samples,
            'A_star_shape': list(self.A_star.shape) if self.A_star is not None else None,
            'b_star_shape': list(self.b_star.shape) if self.b_star is not None else None
        }


def create_eta_wavlm_model(config: Dict[str, Any]) -> EtaWavLMModel:
    """Factory function to create EtaWavLM model"""
    return EtaWavLMModel(config)
