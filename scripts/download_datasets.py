#!/usr/bin/env python3
"""
Audio Dataset Download Script

Downloads and prepares audio datasets for speech processing research.
Supports LibriTTS, LibriSpeech, VCTK, and other common speech datasets.
"""

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve
from datetime import datetime
import os
import sys

import requests
from tqdm import tqdm

# Add project to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_project_root


logger = logging.getLogger(__name__)


# I am not sure what the md5sum of https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 is
# Dataset configurations
DATASET_CONFIGS = {
    "libri-speech": {
        "name": "LibriSpeech",
        "base_url": "https://www.openslr.org/resources/12",
        "splits": {
            "dev-clean": {
                "filename": "dev-clean.tar.gz",
                "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
                "size": "337M",
                "md5": "42e2234ba48799c1f50f24a7926300a1"
            },
            "dev-other": {
                "filename": "dev-other.tar.gz", 
                "url": "https://www.openslr.org/resources/12/dev-other.tar.gz",
                "size": "314M",
                "md5": "c8d0bcc9cca99d4f8b62fcc847357931"
            },
            "test-clean": {
                "filename": "test-clean.tar.gz",
                "url": "https://www.openslr.org/resources/12/test-clean.tar.gz", 
                "size": "346M",
                "md5": "32fa31d27d2e1cad72775fee3f4849a9"
            },
            "test-other": {
                "filename": "test-other.tar.gz",
                "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
                "size": "328M", 
                "md5": "fb5a50374b501bb3bac4815ee91d3135"
            },
            "train-clean-100": {
                "filename": "train-clean-100.tar.gz",
                "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
                "size": "6.3G",
                "md5": "2a93770f6d5c6c964bc36631d331a522"
            },
            "train-clean-360": {
                "filename": "train-clean-360.tar.gz", 
                "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
                "size": "23G",
                "md5": "c0e676e450a7ff2f54aeade5171606fa"
            },
            "train-other-500": {
                "filename": "train-other-500.tar.gz",
                "url": "https://www.openslr.org/resources/12/train-other-500.tar.gz", 
                "size": "30G",
                "md5": "d1a0fd59409feb2c614ce4d30c387708"
            }
        }
    },
    
    "libri-tts": {
        "name": "LibriTTS",
        "base_url": "https://www.openslr.org/resources/60",
        "splits": {
            "dev-clean": {
                "filename": "dev-clean.tar.gz",
                "url": "https://www.openslr.org/resources/60/dev-clean.tar.gz",
                "size": "1.1G",
                "md5": "0c3076c1e5245bb3f0af7d82087ee207"
            },
            "dev-other": {
                "filename": "dev-other.tar.gz",
                "url": "https://www.openslr.org/resources/60/dev-other.tar.gz", 
                "size": "1.1G",
                "md5": "815555d8d75995782ac3ccd7f047213d"
            },
            "test-clean": {
                "filename": "test-clean.tar.gz",
                "url": "https://www.openslr.org/resources/60/test-clean.tar.gz",
                "size": "1.2G", 
                "md5": "7bed3bdb047c4c197f1ad3bc412db59f"
            },
            "test-other": {
                "filename": "test-other.tar.gz",
                "url": "https://www.openslr.org/resources/60/test-other.tar.gz",
                "size": "1.1G",
                "md5": "ae3258249472a13b5abef2a816f733e4"
            },
            "train-clean-100": {
                "filename": "train-clean-100.tar.gz", 
                "url": "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
                "size": "18G",
                "md5": "4a8c202b78fe1bc0c47916a98f3a2ea8"
            },
            "train-clean-360": {
                "filename": "train-clean-360.tar.gz",
                "url": "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
                "size": "65G",
                "md5": "a84ef10ddade5fd25df69596a2767b2d"
            },
            "train-other-500": {
                "filename": "train-other-500.tar.gz",
                "url": "https://www.openslr.org/resources/60/train-other-500.tar.gz",
                "size": "86G", 
                "md5": "7b181dd5ace343a5f38427999684aa6f"
            }
        }
    },
    
    "vctk": {
        "name": "VCTK Corpus",
        "base_url": "https://datashare.ed.ac.uk/handle/10283/3443",
        "splits": {
            "vctk-corpus-0.92": {
                "filename": "VCTK-Corpus-0.92.zip",
                "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
                "size": "10.9G",
                "md5": "8a6ba2946b36fcbef0212cad601f4bfa"
            }
        }
    },
    
    "ljspeech": {
        "name": "LJ Speech", 
        "base_url": "https://keithito.com/LJ-Speech-Dataset",
        "splits": {
            "ljspeech-1.1": {
                "filename": "LJSpeech-1.1.tar.bz2",
                "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                "size": "2.6G",
                "md5": "be1a30275f60de1e9b95d2ee3fbb5091"
            }
        }
    },
    
    "common-voice": {
        "name": "Mozilla Common Voice",
        "base_url": "https://commonvoice.mozilla.org/en/datasets", 
        "note": "Common Voice requires manual download from the website",
        "splits": {}  # Manual download only
    }
}


class DatasetDownloader:
    """Handles downloading and extracting audio datasets"""
    
    def __init__(self, output_dir: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize dataset downloader
        
        Args:
            output_dir: Directory to extract datasets to (default: {project_root}/data)
            cache_dir: Directory to cache downloads (default: {project_root}/cache)
        """
        # Determine project root
        self.project_root = get_project_root()
        logger.info(f"Project root: {self.project_root}")
        
        # Set output and cache directories relative to project root
        if output_dir is None:
            self.output_dir = self.project_root / "data"
        else:
            self.output_dir = Path(output_dir)
            if not self.output_dir.is_absolute():
                self.output_dir = self.project_root / self.output_dir
        
        if cache_dir is None:
            self.cache_dir = self.project_root / "cache"
        else:
            self.cache_dir = Path(cache_dir)
            if not self.cache_dir.is_absolute():
                self.cache_dir = self.project_root / self.cache_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _verify_checksum(self, filepath: Path, expected_md5: str) -> bool:
        """Verify file checksum"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest() == expected_md5
        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False
    
    def download_file_with_progress(self, url: str, filepath: Path, expected_md5: Optional[str] = None) -> bool:
        """
        Download file with progress bar and optional checksum verification
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            expected_md5: Expected MD5 checksum for verification
            
        Returns:
            True if download successful, False otherwise
        """
        # Check if file already exists and is valid
        if filepath.exists():
            if expected_md5 and self._verify_checksum(filepath, expected_md5):
                logger.info(f"File already exists and verified: {filepath.name}")
                return True
            elif not expected_md5:
                logger.info(f"File already exists: {filepath.name}")
                return True
            else:
                logger.warning(f"File exists but checksum invalid, re-downloading: {filepath.name}")
        
        try:
            # Get file size for progress bar
            response = requests.head(url, allow_redirects=True)
            file_size = int(response.headers.get('Content-Length', 0))
            
            # Create progress bar
            with tqdm(
                total=file_size, 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024,
                desc=filepath.name
            ) as pbar:
                
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)
                
                # Download file
                urlretrieve(url, filepath, progress_hook)
            
            # Verify checksum if provided
            if expected_md5:
                logger.info("Verifying checksum...")
                if self._verify_checksum(filepath, expected_md5):
                    logger.info("Checksum verification passed!")
                else:
                    logger.error("Checksum verification failed!")
                    filepath.unlink()  # Remove invalid file
                    return False
            
            logger.info(f"Downloaded: {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive to specified directory"""
        try:
            logger.info(f"Extracting {archive_path.name}...")
            
            if archive_path.suffix.lower() in ['.zip']:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.gz', '.bz2'] or '.tar' in archive_path.name.lower():
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"Unsupported archive format: {archive_path}")
                return False
            
            logger.info(f"Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def _get_file_download_date(self, filepath: Path) -> str:
        """Get the download/creation date of a file, or current date if file doesn't exist"""
        try:
            if filepath.exists():
                # Use file modification time as download date
                timestamp = filepath.stat().st_mtime
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            else:
                # Use current date for new downloads
                return datetime.now().strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Could not get file date: {e}")
            return datetime.now().strftime('%Y-%m-%d')

    def _update_dataset_info(self, dataset_output_dir: Path, dataset_name: str, config: dict, split: str, archive_path: Path):
        """Update dataset info file with new split information"""
        info_path = dataset_output_dir / "dataset_info.json"
        
        # Load existing info or create new
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not read existing info file: {e}, creating new one")
                info = {}
        else:
            info = {}
        
        # Initialize basic info
        info.update({
            "dataset": config['name'],
            "output_dir": str(dataset_output_dir)
        })
        
        # Initialize splits info if not exists
        if "splits_info" not in info:
            info["splits_info"] = {}
        
        # Add/update split information
        download_date = self._get_file_download_date(archive_path)
        info["splits_info"][split] = {
            "download_date": download_date,
            "archive_size": config['splits'][split].get('size', 'unknown')
        }
        
        # Update overall download date to the most recent
        all_dates = [split_info["download_date"] for split_info in info["splits_info"].values()]
        info["last_updated"] = max(all_dates)
        
        # Update splits list
        info["splits"] = list(info["splits_info"].keys())
        
        # Save updated info
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Updated dataset info: {info_path}")

    def download_dataset(self, dataset_name: str, splits: List[str], keep_archives: bool = False) -> bool:
        """
        Download and extract dataset splits
        
        Args:
            dataset_name: Name of dataset to download
            splits: List of splits to download
            keep_archives: Whether to keep archive files after extraction
            
        Returns:
            True if all downloads successful
        """
        if dataset_name not in DATASET_CONFIGS:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
            return False
        
        config = DATASET_CONFIGS[dataset_name]
        logger.info(f"Downloading {config['name']} dataset...")
        
        # Handle special cases
        if dataset_name == "common-voice":
            logger.info("Common Voice requires manual download from: https://commonvoice.mozilla.org/en/datasets")
            return False
        
        dataset_output_dir = self.output_dir / dataset_name.replace("-", "_")
        success = True
        
        for split in splits:
            if split not in config['splits']:
                logger.error(f"Unknown split '{split}' for {config['name']}")
                logger.info(f"Available splits: {list(config['splits'].keys())}")
                success = False
                continue
            
            split_config = config['splits'][split]
            
            logger.info(f"Processing {split} split ({split_config.get('size', 'unknown size')})...")
            
            # Download archive
            archive_path = self.cache_dir / split_config['filename']
            download_success = self.download_file_with_progress(
                split_config['url'],
                archive_path,
                split_config.get('md5')
            )
            
            if not download_success:
                logger.error(f"Failed to download {split}")
                success = False
                continue
            
            # Extract archive
            extract_success = self.extract_archive(archive_path, dataset_output_dir)
            
            if not extract_success:
                logger.error(f"Failed to extract {split}")
                success = False
                continue
            
            # Update dataset info for this split
            self._update_dataset_info(dataset_output_dir, dataset_name, config, split, archive_path)
            
            # Remove archive if requested
            if not keep_archives:
                try:
                    archive_path.unlink()
                    logger.info(f"Removed archive: {archive_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove archive: {e}")
        
        if success:
            logger.info(f"Successfully downloaded {config['name']} dataset to {dataset_output_dir}")
        
        return success
    
    def list_datasets(self):
        """List available datasets and their splits"""
        print("Available datasets:")
        print("=" * 50)
        
        for dataset_name, config in DATASET_CONFIGS.items():
            print(f"\n{config['name']} ({dataset_name}):")
            
            if 'note' in config:
                print(f"  Note: {config['note']}")
            
            if config['splits']:
                print("  Available splits:")
                for split_name, split_config in config['splits'].items():
                    size = split_config.get('size', 'unknown')
                    print(f"    - {split_name} ({size})")
            else:
                print("  Manual download required")
    
    def verify_dataset(self, dataset_name: str) -> bool:
        """Verify that a dataset has been downloaded correctly"""
        dataset_output_dir = self.output_dir / dataset_name.replace("-", "_")
        
        if not dataset_output_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_output_dir}")
            return False
        
        info_path = dataset_output_dir / "dataset_info.json" 
        if not info_path.exists():
            logger.warning(f"Dataset info file not found: {info_path}")
        
        # Basic verification - check if directory contains audio files
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(dataset_output_dir.rglob(f"*{ext}")))
        
        if audio_files:
            logger.info(f"Dataset verification passed: found {len(audio_files)} audio files")
            return True
        else:
            logger.error(f"Dataset verification failed: no audio files found in {dataset_output_dir}")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Download audio datasets for speech processing research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --list
  python download_datasets.py --dataset libri-tts --splits dev-clean test-clean
  python download_datasets.py --dataset libri-speech --splits train-clean-100 dev-clean
  python download_datasets.py --dataset vctk --splits vctk-corpus-0.92
  python download_datasets.py --verify libri-tts
        """
    )
    
    parser.add_argument('--dataset', type=str, help='Dataset to download')
    parser.add_argument('--splits', nargs='+', help='Dataset splits to download')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for datasets (default: {project_root}/data)')
    parser.add_argument('--cache-dir', type=str, 
                       help='Cache directory for downloads (default: {project_root}/cache)')
    parser.add_argument('--keep-archives', action='store_true',
                       help='Keep archive files after extraction')
    parser.add_argument('--list', action='store_true', 
                       help='List available datasets and splits')
    parser.add_argument('--verify', type=str, 
                       help='Verify that a dataset has been downloaded correctly')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create downloader
    downloader = DatasetDownloader(args.output_dir, args.cache_dir)
    
    # Handle different commands
    if args.list:
        downloader.list_datasets()
        return
    
    if args.verify:
        success = downloader.verify_dataset(args.verify)
        exit(0 if success else 1)
    
    if not args.dataset:
        logger.error("No dataset specified. Use --list to see available datasets.")
        parser.print_help()
        exit(1)
    
    if not args.splits:
        logger.error("No splits specified.")
        exit(1)
    
    # Download dataset
    success = downloader.download_dataset(args.dataset, args.splits, args.keep_archives)
    
    if success:
        logger.info("All downloads completed successfully!")
        
        # Verify after download
        if downloader.verify_dataset(args.dataset):
            logger.info("Dataset verification passed!")
        else:
            logger.warning("Dataset verification failed - please check the downloaded files")
    else:
        logger.error("Some downloads failed!")
        exit(1)


if __name__ == "__main__":
    main()
