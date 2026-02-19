"""
Script to download CelebA dataset using torchvision.
"""

import sys
from pathlib import Path


def download_celeba(output_dir='./data'):
    """Download CelebA dataset using torchvision.
    
    Args:
        output_dir: Directory where dataset will be saved
        
    Returns:
        True if successful, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('='*70)
    print('CelebA Dataset Download')
    print('='*70)
    print('\nDownloading CelebA dataset using torchvision...')
    print('This may take a while (the dataset is ~1.4 GB)\n')
    
    try:
        from torchvision.datasets import CelebA
        
        celeba_dir = output_dir / 'celeba_torchvision'
        
        # Download dataset (aligned and cropped version)
        print('Downloading aligned and cropped images...')
        dataset = CelebA(
            root=str(celeba_dir),
            split='all',  # Download all images
            download=True
        )
        
        print(f'\n' + '='*70)
        print('Download Complete!')
        print('='*70)
        print(f"Location: {celeba_dir / 'celeba'}")
        print(f'Number of images: {len(dataset):,}')
        print('\nAdd this path to your config.yaml:')
        print(f"  dataset_path: {celeba_dir / 'celeba' / 'img_align_celeba'}")
        print('='*70)
        
        return True
        
    except ImportError:
        print('\nERROR: torchvision is not installed!')
        print('Install it with: pip install torchvision')
        return False
    except Exception as e:
        error_msg = str(e)
        print(f'\nERROR: Download failed: {e}')
        
        # Check for gdown dependency
        if 'gdown' in error_msg.lower():
            print('\nThe CelebA dataset is hosted on Google Drive, which requires gdown.')
            print('Install it with:')
            print('  pip install gdown')
            print('\nOr install all dependencies:')
            print('  pip install -r requirements.txt')
        else:
            print('\nTroubleshooting:')
            print('  - Check your internet connection')
            print('  - Ensure you have enough disk space (~1.5 GB)')
            print('  - Try running the script again')
        return False


def verify_dataset(dataset_path):
    """Verify that the dataset is properly downloaded.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if valid, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f'ERROR: Dataset path does not exist: {dataset_path}')
        return False
    
    # Count image files
    image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
    num_images = len(image_files)
    
    if num_images == 0:
        print(f'ERROR: No images found in {dataset_path}')
        return False
    
    print(f'\nDataset verification:')
    print(f'  Path: {dataset_path}')
    print(f'  Number of images: {num_images:,}')
    
    if num_images < 100:
        print(f'  WARNING: Very small dataset ({num_images} images)')
        print(f'  CelebA should have ~202,599 images')
        return False
    
    print(f'  Dataset looks good!')
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CelebA dataset')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for dataset (default: ./data)'
    )
    parser.add_argument(
        '--verify',
        type=str,
        default=None,
        help='Verify existing dataset at specified path'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_dataset(args.verify)
        sys.exit(0 if success else 1)
    else:
        success = download_celeba(args.output_dir)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
