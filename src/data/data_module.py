"""
PyTorch Lightning DataModule for Eta-WavLM training

Handles data loading, preprocessing, and batching for the linear transform training.
Since only the linear transform (A*, b*) needs training, this focuses on efficient
feature extraction pipeline.
"""

from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
import torch
import yaml

from .audio_dataset import AudioDataset, collate_fn, load_config

logger = logging.getLogger(__name__)


class EtaWavLMDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Eta-WavLM training
    
    Handles:
    - Generic audio dataset loading via YAML config
    - Train/val/test splits at speaker level (important for generalization)
    - Efficient batching for feature extraction
    - Data caching and memory management
    """
    
    def __init__(
        self,
        config_path: str,
        stage_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EtaWavLM DataModule
        
        Args:
            config_path: Path to YAML configuration file
            stage_overrides: Dictionary to override config parameters
        """
        super().__init__()
        
        # Load configuration
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Apply overrides
        if stage_overrides:
            self.config.update(stage_overrides)
        
        # Store for Lightning's hyperparameter saving
        self.save_hyperparameters({"config_path": config_path, "stage_overrides": stage_overrides})
        
        # Extract commonly used parameters
        self.data_dir = Path(self.config['data_dir'])
        self.batch_size = self.config.get('batch_size', 16)
        self.eval_batch_size = self.config.get('eval_batch_size', self.batch_size)
        self.num_workers = self.config.get('num_workers', 4)
        self.pin_memory = self.config.get('pin_memory', True)
        self.persistent_workers = self.config.get('persistent_workers', True)
        
        # Dataset containers
        self.train_dataset: Optional[AudioDataset] = None
        self.val_dataset: Optional[AudioDataset] = None
        self.test_dataset: Optional[AudioDataset] = None
        
        # Dataset statistics
        self.dataset_stats = {}

    def prepare_data(self) -> None:
        """
        Download and prepare data (called once per node)

        Verify the data exists at specified paths.
        """
        super().prepare_data()

        if not self.data_dir.exists():
            logger.warning(f"Data directory not found at {self.data_dir}")
            logger.info("Please ensure your data is available at the configured path")
            return

        # Check for required splits - handle train_split as list or string
        train_split = self.config.get('train_split', 'train')
        val_split = self.config.get('val_split', 'val')
        test_split = self.config.get('test_split', 'test')

        # Flatten train_split if it's a list
        if isinstance(train_split, list):
            all_splits = train_split + [val_split, test_split]
        else:
            all_splits = [train_split, val_split, test_split]

        # Try to find split directories (if they exist as separate folders)
        for split_name in all_splits:
            possible_paths = [
                self.data_dir / split_name,
                self.data_dir / "LibriSpeech" / split_name,
            ]
            found = any(p.exists() for p in possible_paths)
            if not found:
                logger.debug(f"Split {split_name} directory not found (may be in single directory)")
                            
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage (called on each GPU/node)

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        super().setup(stage)

        # Set seed for reproducibility
        torch.manual_seed(42)

        if stage == "fit" or stage is None:
            if hasattr(self, '_setup_fit_done') and self._setup_fit_done:
                return  # Skip if already set up
        
            # Create training dataset(s) - handle multiple splits
            train_splits = self.config.get('train_split', 'train-clean-100')
            if isinstance(train_splits, str):
                train_splits = [train_splits]

            # Create and combine multiple training datasets
            train_datasets = []
            total_samples = 0
            total_speakers = set()

            for split in train_splits:
                logger.info(f"Loading training split: {split}")

                # Create config copy with specific split
                split_config = self.config.copy()
                split_config['train_split'] = split  # Set specific split

                dataset = AudioDataset(
                    config=split_config,
                    split="train"  # Still use "train" as the mode
                )

                train_datasets.append(dataset)
                split_stats = dataset.get_dataset_stats()
                split_speakers = dataset.get_speaker_ids() if hasattr(dataset, 'get_speaker_ids') else set()

                total_samples += len(dataset)
                total_speakers.update(split_speakers)

                logger.info(f"  {split}: {len(dataset)} samples, {len(split_speakers)} speakers")

            # Combine all training datasets
            if len(train_datasets) == 1:
                self.train_dataset = train_datasets[0]
            else:
                from torch.utils.data import ConcatDataset
                self.train_dataset = ConcatDataset(train_datasets)

            logger.info(f"Combined training: {total_samples} samples, {len(total_speakers)} speakers")

            # Create validation dataset
            val_split = self.config.get('val_split', 'dev-clean')
            val_config = self.config.copy()
            val_config['val_split'] = val_split

            self.val_dataset = AudioDataset(
                config=val_config,
                split="val"
            )

            # Log dataset info
            val_stats = self.val_dataset.get_dataset_stats()
            logger.info(f"Validation: {len(self.val_dataset)} samples, {val_stats.get('num_speakers', 'N/A')} speakers")

            self.dataset_stats['train'] = {
                'total_samples': total_samples,
                'num_speakers': len(total_speakers),
                'splits': train_splits
            }
            self.dataset_stats['val'] = val_stats

            self._setup_fit_done = True

        if stage == "test" or stage is None:
            if hasattr(self, '_setup_test_done') and self._setup_test_done:
                return  # Skip if already set up
        
            # Create test dataset
            test_split = self.config.get('test_split', 'test-clean')
            test_config = self.config.copy()
            test_config['test_split'] = test_split

            self.test_dataset = AudioDataset(
                config=test_config,
                split="test"
            )

            test_stats = self.test_dataset.get_dataset_stats()
            logger.info(f"Test: {len(self.test_dataset)} samples, {test_stats.get('num_speakers', 'N/A')} speakers")

            self.dataset_stats['test'] = test_stats

            self._setup_test_done = True
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_fn,
            drop_last=self.config.get('drop_last', False)
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (uses test data)"""
        return self.test_dataloader()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Get class weights for speaker balancing (if needed)
        
        For speaker classification evaluation, we might want to balance speakers
        """
        if not self.train_dataset:
            raise RuntimeError("Must setup datasets first")
        
        # Count samples per speaker
        speaker_counts = {}
        for item in self.train_dataset.data_list:
            speaker_id = item['speaker_id']
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        # Calculate inverse frequency weights
        total_samples = sum(speaker_counts.values())
        num_speakers = len(speaker_counts)
        
        weights = {}
        for speaker_id, count in speaker_counts.items():
            weights[speaker_id] = total_samples / (num_speakers * count)
        
        # Convert to tensor indexed by speaker ID
        max_speaker_id = max(speaker_counts.keys())
        weight_tensor = torch.ones(max_speaker_id + 1)
        
        for speaker_id, weight in weights.items():
            weight_tensor[speaker_id] = weight
        
        return weight_tensor
    
    def get_speaker_mapping(self) -> Dict[int, int]:
        """
        Get mapping from original speaker IDs to contiguous indices
        
        Useful for speaker classification tasks
        """
        if not self.train_dataset:
            raise RuntimeError("Must setup datasets first")
        
        unique_speakers = sorted(self.train_dataset.get_speaker_ids())
        return {speaker_id: idx for idx, speaker_id in enumerate(unique_speakers)}
    
    def create_speaker_subset(self) -> 'EtaWavLMDataModule':
        """
        Create a subset with limited number of speakers for evaluation
        
        Uses speaker_subset configuration from config file.
        
        Returns:
            New DataModule with speaker subset
        """
        if not self.train_dataset:
            raise RuntimeError("Must setup datasets first")
        
        # Create subset datasets using config
        train_subset = self.train_dataset.create_speaker_subset()
        
        # Create validation subset if validation dataset exists
        val_subset = None
        if self.val_dataset:
            val_subset = self.val_dataset.create_speaker_subset()
        
        # Create new DataModule with same config
        subset_dm = EtaWavLMDataModule(
            config_path=self.config_path,
            stage_overrides=self.hparams.get('stage_overrides')
        )
        
        # Assign subset datasets
        subset_dm.train_dataset = train_subset
        subset_dm.val_dataset = val_subset
        subset_dm.test_dataset = self.test_dataset  # Keep full test set
        
        # Update config for subset
        subset_dm.config = self.config.copy()
        
        subset_config = self.config.get('speaker_subset', {})
        n_speakers = subset_config.get('n_speakers', 10)
        
        logger.info(f"Created speaker subset: {n_speakers} speakers")
        logger.info(f"Train subset: {len(train_subset)} samples")
        if val_subset:
            logger.info(f"Val subset: {len(val_subset)} samples")
        
        return subset_dm
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration parameters"""
        self.config.update(updates)
        
        # Update commonly used parameters
        self.batch_size = self.config.get('batch_size', 16)
        self.eval_batch_size = self.config.get('eval_batch_size', self.batch_size)
        self.num_workers = self.config.get('num_workers', 4)
        self.pin_memory = self.config.get('pin_memory', True)
        self.persistent_workers = self.config.get('persistent_workers', True)


def create_datamodule_from_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None
) -> EtaWavLMDataModule:
    """
    Create DataModule from configuration file
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Optional parameter overrides
        
    Returns:
        Configured EtaWavLMDataModule
    """
    return EtaWavLMDataModule(
        config_path=config_path,
        stage_overrides=overrides
    )


if __name__ == "__main__":
    # Test the DataModule
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test config
    test_config = {
        'data_dir': './data/librispeech',
        'train_split': 'test-clean',
        'val_split': 'test-clean', 
        'test_split': 'test-clean',
        'sample_rate': 16000,
        'max_duration': 10.0,
        'min_duration': 1.0,
        'max_train_samples': 20,
        'max_val_samples': 10,
        'max_test_samples': 10,
        'batch_size': 4,
        'eval_batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'speaker_subset': {
            'n_speakers': 3,
            'min_utterances_per_speaker': 2,
            'max_utterances_per_speaker': 5
        }
    }
    
    # Save test config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        test_config_path = f.name
    
    try:
        # Create DataModule
        dm = EtaWavLMDataModule(
            config_path=test_config_path,
            stage_overrides={'batch_size': 2}  # Override batch size
        )
        
        # Setup
        dm.prepare_data()
        dm.setup("fit")
        
        if dm.train_dataset and len(dm.train_dataset) > 0:
            # Test dataloaders
            train_dl = dm.train_dataloader()
            val_dl = dm.val_dataloader()
            
            print(f"Train batches: {len(train_dl)}")
            print(f"Val batches: {len(val_dl)}")
            
            # Test batch
            batch = next(iter(train_dl))
            print(f"Batch keys: {batch.keys()}")
            print(f"Waveforms shape: {batch['waveforms'].shape}")
            print(f"Speaker IDs: {batch['speaker_ids']}")
            
            # Test speaker subset
            subset_dm = dm.create_speaker_subset()
            print(f"Subset train size: {len(subset_dm.train_dataset)}")
        else:
            print("No data found - this is expected if LibriSpeech is not available")
    
    except Exception as e:
        print(f"Test failed (expected if no data): {e}")
    
    finally:
        # Cleanup
        import os
        os.unlink(test_config_path)
