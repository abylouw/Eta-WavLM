#!/usr/bin/env python3
"""
Training Script for Eta-WavLM

Complete training pipeline from configuration to trained model.
Handles data loading, model initialization, training, and evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_project_root, load_config, load_training_config
from src.data import create_datamodule_from_config
from src.models import create_eta_wavlm_lightning_module
from src.features import verify_speaker_encoder

logger = logging.getLogger(__name__)


def verify_requirements(config: Dict[str, Any]) -> bool:
    """Verify that required models and data are available"""
    logger.info("Verifying requirements...")
    
    # Check data directory
    project_root = get_project_root()
    data_dir = project_root / config.get('data_dir', 'data')
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run: python scripts/download_datasets.py --help")
        return False
    
    # Check SSL model (will auto-download if needed)
    ssl_model_name = config.get('ssl_model', {}).get('name', 'microsoft/wavlm-large')
    logger.info(f"SSL model: {ssl_model_name}")
    
    # Check speaker model
    speaker_source = config.get('speaker_model', {}).get('source', 'speechbrain/spkrec-ecapa-voxceleb')
    logger.info(f"Speaker model: {speaker_source}")
    
    # Verify speaker model can be loaded
    try:
        if not verify_speaker_encoder(speaker_source):
            logger.warning(f"Speaker model verification failed: {speaker_source}")
            logger.info("Run: python scripts/download_models.py --help")
    except Exception as e:
        logger.warning(f"Could not verify speaker model: {e}")
        
    return True


def setup_logging_and_checkpoints(
        config: Dict[str, Any], 
        output_dir: Path,
        experiment_name: str,
        stage: Optional[int] = None
) -> tuple:
    """Setup logging and checkpointing for training - stage-aware version"""
    project_root = get_project_root()
    
    # Resolve output_dir relative to project root
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
        
    # Get logging config
    logging_config = config.get('logging', {})
    logging_default = logging_config.get('default', {'name': 'tensorboard'})
    logging_impl = logging_config.get(logging_default.get('name', 'tensorboard'), {})
    logs_base_dir = project_root / logging_impl.get('save_dir', 'logs')
    
    # Create loggers
    tb_logger = TensorBoardLogger(
        save_dir=str(logs_base_dir),
        name=experiment_name,
        version=None
    )
    
    csv_logger = CSVLogger(
        save_dir=str(logs_base_dir),
        name=experiment_name
    )
    
    # Create callbacks
    callbacks = []
    
    # Model checkpointing - stage-aware
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_dir = project_root / checkpoint_config.get('dirpath', 'checkpoints')
    
    # Detect stage if not provided
    if stage is None:
        stage = config.get('training', {}).get('stage', 2)  # Default to stage 2
        
    if stage == 1:
        # Stage 1: Monitor PCA explained variance
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{experiment_name}_stage1_" + "{epoch:02d}-{stage1/explained_variance:.3f}",
            monitor="stage1/explained_variance",
            mode="max",  # Higher explained variance is better
            save_top_k=checkpoint_config.get('save_top_k', 1),
            save_last=checkpoint_config.get('save_last', True),
            auto_insert_metric_name=False
        )
    else:  
        # Stage 2: Monitor solved status instead of validation metric
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{experiment_name}_stage2_" + "{epoch:02d}-{stage2/solved:.0f}",
            monitor="stage2/solved",
            mode="max",  # 1.0 when solved, 0.0 when not
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False
        )
        
    callbacks.append(checkpoint_callback)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return [tb_logger, csv_logger], callbacks


def create_trainer(
        config: Dict[str, Any],
        loggers: list = None,
        callbacks: list = None,
        max_epochs: int = 1,
        enable_logging: bool = True
) -> pl.Trainer:
    """Create Lightning trainer with configuration"""
    # Extract training configuration

    trainer_config = config.get('training', {})
    
    # Determine devices
    if torch.cuda.is_available():
        devices = trainer_config.get('devices', 'auto')
        accelerator = 'gpu'
    else:
        devices = 1
        accelerator = 'cpu'

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        logger=loggers if enable_logging else False,
        callbacks=callbacks or [],
        enable_checkpointing=enable_logging,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        max_steps=-1,
        limit_train_batches=trainer_config.get('limit_train_batches', 1.0),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
        precision=trainer_config.get('precision', 32)
    )
    
    return trainer


def save_results(
        model: pl.LightningModule,
        output_dir: Path,
        experiment_name: str,
        config: Dict[str, Any]
):
    """Save training results and artifacts"""
    project_root = get_project_root()

    # Resolve output_dir relative to project root
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters
    model_path = output_dir / f"{experiment_name}_model.pt"
    model.save_model(str(model_path))

    # Save config
    config_path = output_dir / f"{experiment_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # Save model info
    info_path = output_dir / f"{experiment_name}_info.json"
    model_info = model.get_model_info()
    
    import json
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
        
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Info: {info_path}")

    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Model: {output_dir / f'{experiment_name}_model.pt'}")
    logger.info(f"Config: {output_dir / f'{experiment_name}_config.yaml'}")
    logger.info(f"Info: {output_dir / f'{experiment_name}_info.json'}")


def run_two_stage_training(model, datamodule, config, output_dir, experiment_name):
    """Run complete two-stage training pipeline"""
    project_root = get_project_root()
    
    # Determine stage paths
    training_config = config.get('training', {})
    embeddings_save_path = project_root / training_config.get('embeddings_save_path', 'speaker_embeddings.pkl')
    pca_save_path = project_root / training_config.get('pca_save_path', 'pca_model.pkl')
    
    # STAGE 1: Speaker embeddings and PCA
    stage1_needed = not (embeddings_save_path.exists() and pca_save_path.exists())
    
    if stage1_needed:
        logger.info("=== STAGE 1: SPEAKER EMBEDDINGS & PCA ===")
        
        # Create stage 1 model and datamodule
        stage1_config = config.copy()
        stage1_config['training']['stage'] = 1
        
        stage1_model = create_eta_wavlm_lightning_module(stage1_config)
        stage1_loggers, stage1_callbacks = setup_logging_and_checkpoints(
            config, output_dir, experiment_name, stage=1
        )
        max_epochs = stage1_config.get('training', {}).get('max_epochs', 1)
        stage1_trainer = create_trainer(config, stage1_loggers, stage1_callbacks, max_epochs)
        
        # Train stage 1
        stage1_trainer.fit(stage1_model, datamodule)
        
        logger.info("Stage 1 completed - speaker embeddings and PCA saved")
        
        # Update model reference for stage 2
        model = stage1_model
        
    else:
        logger.info("=== STAGE 1: OUTPUTS EXIST - SKIPPING ===")
        logger.info(f"Found: {embeddings_save_path}")
        logger.info(f"Found: {pca_save_path}")
        
    # STAGE 2: SSL extraction and decomposition solving
    logger.info("=== STAGE 2: SSL EXTRACTION & DECOMPOSITION ===")
    
    # Create stage 2 model and trainer
    stage2_config = config.copy()
    stage2_config['training']['stage'] = 2
    
    # Use existing model if we ran stage 1, otherwise create new one
    if not stage1_needed:
        model = create_eta_wavlm_lightning_module(stage2_config)
    else:
        # Update existing model's stage
        model.current_stage = 2
        model.config = stage2_config
        
    stage2_loggers, stage2_callbacks = setup_logging_and_checkpoints(
        config, output_dir, experiment_name, stage=2
    )
    stage2_trainer = create_trainer(config, stage2_loggers, stage2_callbacks)
    
    # Train stage 2 (online accumulation and solving)
    stage2_trainer.fit(model, datamodule)
    
    # Check if decomposition was solved
    if model.model.parameters_solved:
        logger.info("=== TWO-STAGE TRAINING COMPLETED ===")
        return True, model
    else:
        logger.error("++ Stage 2 failed to solve decomposition ++")
        return False, model


def run_validation(model, datamodule, config, enable_logging=True):
    """Run validation on trained model"""
    logger.info("Running validation...")
    
    # Create simple trainer for validation
    trainer = create_trainer(config, enable_logging=enable_logging)
    
    try:
        # Check if datamodule has validation data
        datamodule.setup('fit')
        val_loader = datamodule.val_dataloader()
        
        if val_loader is None:
            logger.warning("No validation data available - skipping validation")
            return True
        
        trainer.validate(model, datamodule)
        logger.info("-- Validation completed --")
        return True
    
    except Exception as e:
        logger.error(f"++ Validation failed: {e} ++")
        logger.warning("You can use the evaluation script instead: python scripts/evaluate_model.py")
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description="Train Eta-WavLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal training
  python train.py --config configs/train.yaml
  
  # Validation only
  python train.py --config configs/train.yaml --validate-only --model-path path/to/model.pt
  
  # Custom experiment
  python train.py --config configs/train.yaml --experiment-name my_experiment --output-dir outputs/custom
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, required=True,
                        help='Training configuration file (e.g., configs/train.yaml)')
    
    # Training modes
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation on trained model')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model for validation')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Maximum training epochs (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (overrides config)')
    
    # Options
    parser.add_argument('--skip-verification', action='store_true',
                        help='Skip requirement verification')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation after training')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (limited data)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Load training configuration
        config = load_training_config(args.config)
        logger.info(f"Loaded training config: {args.config}")
        
        # Apply command line overrides
        if args.max_epochs is not None:
            config.setdefault('training', {})['max_epochs'] = args.max_epochs
            
        if args.output_dir is not None:
            config.setdefault('output', {})['base_dir'] = args.output_dir
            
        if args.experiment_name is not None:
            config['experiment_name'] = args.experiment_name
            
        # Debug mode - limit data
        if args.debug:
            logger.info("Debug mode enabled - limiting data")
            config['max_train_samples'] = 50
            config['max_val_samples'] = 20
            config['max_test_samples'] = 20

        # Setup experiment
        experiment_name = config.get('experiment_name', 'eta_wavlm_experiment')
        if not experiment_name or experiment_name == 'eta_wavlm_experiment':
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"eta_wavlm_{timestamp}"
            
        # Setup output directory
        project_root = get_project_root()
        output_base = config.get('output', {}).get('base_dir', 'outputs')
        output_dir = project_root / output_base / experiment_name
        
        logger.info(f"Starting Eta-WavLM experiment: {experiment_name}")
        logger.info(f"Output directory: {output_dir}")
        
        # Verify requirements
        if not args.skip_verification:
            if not verify_requirements(config):
                logger.error("Requirement verification failed")
                sys.exit(1)
                
        # Create data module and model
        logger.info("Creating data module...")
        datamodule = create_datamodule_from_config(args.config, config)
        
        # Determine execution mode
        if args.validate_only:
            # VALIDATION ONLY MODE
            if not args.model_path:
                logger.error("--model-path required for validation-only mode")
                sys.exit(1)
                
            logger.info("=== VALIDATION ONLY MODE ===")
            model = create_eta_wavlm_lightning_module(config)
            model.load_model(args.model_path)
            validation_success = run_validation(model, datamodule, config)
            sys.exit(0 if validation_success else 1)
            
        else:
            # TWO-STAGE TRAINING MODE
            logger.info("=== TWO-STAGE TRAINING MODE ===")
            
            # Create initial model
            model = create_eta_wavlm_lightning_module(config)
            
            # Run two-stage training (handles logging and checkpoints internally)
            training_success, model = run_two_stage_training(model, datamodule, config, output_dir, experiment_name)
            
        # Save results if training succeeded
        if training_success:
            save_results(model, output_dir, experiment_name, config)
            
            # Run validation unless skipped
            if not args.skip_validation:
                logger.info("=== POST-TRAINING VALIDATION ===")
                run_validation(model, datamodule, config, enable_logging=False)
                
            # Print final results
            model_info = model.get_model_info()
            eta_info = model_info.get('eta_wavlm', {})
            
            logger.info("=== FINAL RESULTS ===")
            logger.info(f"  Samples processed: {eta_info.get('n_accumulated_samples', 0)}")
            logger.info(f"  Parameters solved: {eta_info.get('parameters_solved', False)}")
            logger.info(f"  SSL dim: {eta_info.get('ssl_dim', 0)}")
            logger.info(f"  Speaker dim: {eta_info.get('speaker_dim', 0)}")
            logger.info("-- Training completed successfully! -- ")
            
        else:
            logger.error("++ Training failed ++")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
