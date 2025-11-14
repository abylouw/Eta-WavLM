"""
Unified Lightning Module for EtaWavLM Training

Stage 1: Collect speaker embeddings, fit PCA, save results
Stage 2: Online SSL extraction with matrix accumulation, solve decomposition
"""

from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

from .eta_wavlm_model import EtaWavLMModel, create_eta_wavlm_model

logger = logging.getLogger(__name__)


class EtaWavLMLightningModule(pl.LightningModule):
    """
    Unified Lightning Module for EtaWavLM Training
    
    Stage 1: Collect speaker embeddings, fit PCA, save results
    Stage 2: Online SSL extraction with matrix accumulation, solve decomposition
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        # Training configuration
        training_config = config.get('training', {})
        self.current_stage = training_config.get('stage', 1)
        self.embeddings_save_path = training_config.get('embeddings_save_path', 'speaker_embeddings.pkl')
        self.pca_save_path = training_config.get('pca_save_path', 'pca_model.pkl')
        
        # Evaluation configuration
        self.evaluation_config = config.get('evaluation', {})
        
        # Create model
        self.model = create_eta_wavlm_model(config)
        
        # Training state
        self.epoch_started = False
        
        # Validation storage (for Stage 2)
        self.validation_ssl_features = []
        self.validation_eta_features = []
        self.validation_speaker_ids = []
        
        logger.info(f"Initialized Lightning Module for Stage {self.current_stage}")
    
    def on_train_epoch_start(self):
        """Start the appropriate training stage"""
        logger.info(f"Starting training epoch {self.current_epoch} - Stage {self.current_stage}")
        
        if self.current_stage == 1:
            # Check if Stage 1 outputs already exist
            if Path(self.embeddings_save_path).exists() and Path(self.pca_save_path).exists():
                logger.info("Stage 1 outputs already exist - loading existing results")
                # Load existing results without redoing work
                return
            
            # Start Stage 1: collecting speaker embeddings
            self.model.start_stage1()
            logger.info("Stage 1: Collecting speaker embeddings for PCA fitting")
            
        elif self.current_stage == 2:
            # Check prerequisites first
            if not Path(self.embeddings_save_path).exists():
                raise RuntimeError(f"Speaker embeddings not found: {self.embeddings_save_path}. Run Stage 1 first.")
            if not Path(self.pca_save_path).exists():
                raise RuntimeError(f"PCA model not found: {self.pca_save_path}. Run Stage 1 first.")
            
            # Start Stage 2: online SSL extraction and accumulation
            self.model.start_stage2(self.embeddings_save_path)
            logger.info("Stage 2: Starting online SSL extraction and accumulation")
            
        else:
            raise ValueError(f"Invalid stage: {self.current_stage}")
        
        self.epoch_started = True
        
        # Clear validation storage
        self.validation_ssl_features.clear()
        self.validation_eta_features.clear()
        self.validation_speaker_ids.clear()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - stage-specific processing"""
        waveforms = batch['waveforms']
        lengths = batch.get('lengths')
        
        if self.current_stage == 1:
            # Stage 1: Collect speaker embeddings
            stats = self.model.collect_speaker_embeddings_stage1(waveforms, lengths, batch_idx)

            self.log('stage1/batch_size', stats['batch_size'])
            self.log('stage1/total_utterances', stats['total_utterances_collected'])
            
            # Log progress occasionally
            if batch_idx % 50 == 0:
                logger.info(f"Stage 1 - Batch {batch_idx}: collected {stats['total_utterances_collected']} utterances")
            
        elif self.current_stage == 2:
            # Stage 2: Extract SSL and update accumulators online
            stats = self.model.collect_ssl_features_stage2(waveforms, lengths, batch_idx)
                        
            self.log('stage2/batch_size', stats['batch_size'])
            self.log('stage2/total_ssl_collected', stats['total_ssl_collected'])
            self.log('stage2/utterances_processed', stats['utterances_processed'])
            
            # Log progress occasionally
            if batch_idx % 50 == 0:
                logger.info(f"Stage 2 - Batch {batch_idx}: accumulated {stats['total_ssl_collected']} samples "
                           f"from {stats['utterances_processed']} utterances")
        
        # Return dummy loss
        return torch.tensor(0.0, requires_grad=True)
    
    def on_train_epoch_end(self):
        """End of training epoch - stage-specific completion"""
        if not self.epoch_started:
            return
        
        if self.current_stage == 1:
            # Check if we already had the outputs
            if Path(self.embeddings_save_path).exists() and Path(self.pca_save_path).exists():
                logger.info("Stage 1 outputs already existed - skipped collection")
                return
            
            # Stage 1: Fit PCA and save results
            logger.info("Stage 1 complete - fitting PCA and saving results...")
            
            # Fit PCA on collected embeddings and save everything
            pca_stats = self.model.fit_pca_and_save_stage1()
            
            # Log statistics
            self.log('stage1/num_utterances', pca_stats['num_utterances'])
            self.log('stage1/pca_components', pca_stats['pca_components'])
            self.log('stage1/explained_variance', pca_stats['total_explained_variance'])
            
            logger.info(f"Stage 1 complete!")
            logger.info(f"  Processed {pca_stats['num_utterances']} utterances")
            logger.info(f"  PCA explained variance: {pca_stats['total_explained_variance']:.3f}")
            logger.info(f"  Saved to: {pca_stats['embeddings_path']}, {pca_stats['pca_path']}")
            logger.info(f"Next: Run Stage 2 with the same data to solve decomposition")
            
        elif self.current_stage == 2:
            # Stage 2: Solve decomposition from accumulated matrices
            logger.info("Stage 2 complete - solving decomposition from accumulated data...")
            
            # Solve decomposition directly from accumulated matrices
            solution_stats = self.model.solve_decomposition_stage2()
            
            if solution_stats['success']:
                self.log('stage2/solved', 1.0)
                self.log('stage2/condition_number', solution_stats['condition_number'])
                self.log('stage2/residual_norm', solution_stats['residual_norm'])
                self.log('stage2/n_samples', solution_stats['n_samples'])
                self.log('stage2/n_utterances', solution_stats['n_utterances'])
                
                logger.info("Stage 2 complete! Linear decomposition solved successfully.")
                logger.info(f"  A* shape: {solution_stats['A_star_shape']}")
                logger.info(f"  b* shape: {solution_stats['b_star_shape']}")
                logger.info(f"  Condition number: {solution_stats['condition_number']:.2e}")
                logger.info(f"  Residual norm: {solution_stats['residual_norm']:.2e}")
                logger.info(f"  Processed {solution_stats['n_utterances']} utterances, {solution_stats['n_samples']} total samples")
                
            else:
                self.log('stage2/solved', 0.0)
                logger.error(f"Stage 2 failed: {solution_stats.get('error', 'Unknown error')}")
        
        self.epoch_started = False

    def on_validation_epoch_start(self):
        """Select random subset of speakers for evaluation"""
        if self.current_stage != 2 or not self.model.parameters_solved:
            return

        # Get evaluation config
        eval_config = self.evaluation_config.get('speaker_classification', {})
        max_speakers = eval_config.get('max_speakers', 10)

        # Get all unique speaker IDs from the validation dataset
        try:
            val_dataset = self.trainer.datamodule.val_dataset
            all_speaker_ids = torch.unique(torch.tensor([sample['speaker_id'] for sample in val_dataset]))

            # Randomly select subset
            if len(all_speaker_ids) > max_speakers:
                perm = torch.randperm(len(all_speaker_ids))[:max_speakers]
                self.selected_speaker_ids = all_speaker_ids[perm].tolist()
            else:
                self.selected_speaker_ids = all_speaker_ids.tolist()

            logger.info(f"Selected {len(self.selected_speaker_ids)} speakers for evaluation: {self.selected_speaker_ids}")
        except Exception as e:
            logger.warning(f"Could not select speakers for evaluation: {e}")
            self.selected_speaker_ids = []

        # Clear validation storage
        self.validation_ssl_features.clear()
        self.validation_eta_features.clear()
        self.validation_speaker_ids.clear()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step - only process selected speakers"""
        if self.current_stage != 2 or not self.model.parameters_solved:
            return {'val_loss': torch.tensor(0.0)}

        # Skip if no selected speakers
        if not hasattr(self, 'selected_speaker_ids') or not self.selected_speaker_ids:
            return {'val_loss': torch.tensor(0.0)}

        waveforms = batch['waveforms']
        lengths = batch.get('lengths')
        speaker_ids = batch['speaker_ids']

        # Filter to only selected speakers - fix device mismatch
        selected_tensor = torch.tensor(self.selected_speaker_ids, device=speaker_ids.device)
        mask = torch.isin(speaker_ids, selected_tensor)
    
        if not mask.any():
            return {'val_loss': torch.tensor(0.0)}  # Skip this batch

        # Process only selected speakers
        waveforms = waveforms[mask]
        speaker_ids = speaker_ids[mask]
        if lengths is not None:
            lengths = lengths[mask]
                
        # Generate eta representations
        with torch.no_grad():
            components = self.model.generate_eta_representations(
                waveforms, lengths, speaker_ids=speaker_ids, return_components=True
            )
            
            eta_reps = components['eta']
            ssl_features = components['ssl_features']
            speaker_ids_expanded = components['speaker_ids_expanded']

        # Store as numpy for memory efficiency
        self.validation_eta_features.append(eta_reps.cpu().numpy())
        self.validation_ssl_features.append(ssl_features.cpu().numpy())
        self.validation_speaker_ids.append(speaker_ids_expanded.cpu().numpy())
        
        # Free GPU memory
        del components, eta_reps, ssl_features, speaker_ids_expanded
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'val_loss': torch.tensor(0.0),
            'eta_mean': torch.mean(eta_reps) if 'eta_reps' in locals() else 0.0,
            'eta_std': torch.std(eta_reps) if 'eta_reps' in locals() else 0.0
        }
    
    def on_validation_epoch_end(self):
        """Evaluate speaker independence after validation"""
        if (self.current_stage != 2 or 
            not self.model.parameters_solved or 
            not self.validation_eta_features):
            return
        
        logger.info("Evaluating speaker independence...")
        
        # Concatenate all validation features
        all_eta_features = np.concatenate(self.validation_eta_features, axis=0)
        all_ssl_features = np.concatenate(self.validation_ssl_features, axis=0)
        all_speaker_ids = np.concatenate(self.validation_speaker_ids, axis=0)
        
        logger.info(f"Evaluation data: {all_eta_features.shape[0]} samples, "
                   f"{len(np.unique(all_speaker_ids))} speakers")
        
        # Speaker classification evaluation
        try:
            # Evaluate eta representations
            eta_accuracy = self._evaluate_speaker_classification(
                all_eta_features, all_speaker_ids, "Eta"
            )
            
            # Evaluate original SSL features for comparison
            ssl_accuracy = self._evaluate_speaker_classification(
                all_ssl_features, all_speaker_ids, "SSL"
            )
            
            # Log results
            self.log('val_eta_speaker_accuracy', eta_accuracy)
            self.log('val_ssl_speaker_accuracy', ssl_accuracy)
            self.log('val_speaker_accuracy_reduction', ssl_accuracy - eta_accuracy)
            
            logger.info(f"Speaker Classification Results:")
            logger.info(f"  SSL features accuracy: {ssl_accuracy:.1f}%")
            logger.info(f"  Eta features accuracy: {eta_accuracy:.1f}%")
            logger.info(f"  Accuracy reduction: {ssl_accuracy - eta_accuracy:.1f}%")
            
        except Exception as e:
            logger.error(f"Speaker classification evaluation failed: {e}")
        
        # Clear validation storage
        self.validation_ssl_features.clear()
        self.validation_eta_features.clear()
        self.validation_speaker_ids.clear()
    
    def _evaluate_speaker_classification(
        self, 
        features: np.ndarray, 
        speaker_ids: np.ndarray,
        feature_name: str
    ) -> float:
        """Evaluate speaker classification accuracy"""
        
        # Subsample if too many samples
        max_samples = 10000
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            speaker_ids = speaker_ids[indices]
            logger.info(f"Subsampled {max_samples} samples for {feature_name} evaluation")
        
        try:
            classifier = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=1000)
            
            # Use 5-fold cross-validation
            n_speakers = len(np.unique(speaker_ids))
            cv_folds = min(5, n_speakers)
            
            cv_scores = cross_val_score(
                classifier, features, speaker_ids, 
                cv=cv_folds, 
                scoring='accuracy'
            )
            
            accuracy = np.mean(cv_scores) * 100
            
            logger.info(f"{feature_name} classification accuracy: {accuracy:.1f}% (±{np.std(cv_scores)*100:.1f}%)")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Classification failed for {feature_name}: {e}")
            return 0.0
    
    def configure_optimizers(self):
        """Dummy optimizer for Lightning"""
        return torch.optim.Adam([torch.tensor(0.0, requires_grad=True)], lr=1e-3)
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save_parameters(path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model.load_parameters(path)
        logger.info(f"Loaded model from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        model_info = self.model.get_model_info()
        return {
            'eta_wavlm': model_info,
            'current_stage': self.current_stage,
            'embeddings_save_path': self.embeddings_save_path,
            'pca_save_path': self.pca_save_path
        }


def create_eta_wavlm_lightning_module(config: Dict[str, Any]) -> EtaWavLMLightningModule:
    """Factory function to create Lightning module"""
    return EtaWavLMLightningModule(config)
