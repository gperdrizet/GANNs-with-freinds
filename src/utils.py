"""
Utility functions for distributed GAN training.
"""

import os
import socket
import uuid
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torchvision.utils as vutils
from PIL import Image


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_db_url(db_config: Dict[str, Any]) -> str:
    """Build PostgreSQL connection URL from config.
    
    Args:
        db_config: Database configuration dictionary
        
    Returns:
        PostgreSQL connection URL
    """
    return (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    return device


def get_gpu_name() -> str:
    """Get GPU name if available.
    
    Returns:
        GPU name or 'CPU'
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return 'CPU'


def get_hostname() -> str:
    """Get machine hostname.
    
    Returns:
        Hostname
    """
    return socket.gethostname()


def generate_worker_id() -> str:
    """Generate unique worker ID.
    
    Returns:
        Unique worker identifier
    """
    return f'{get_hostname()}_{uuid.uuid4().hex[:8]}'


def save_generated_images(
    images: torch.Tensor,
    output_path: str,
    nrow: int = 8,
    normalize: bool = True
):
    """Save a grid of generated images.
    
    Args:
        images: Tensor of images (N, C, H, W)
        output_path: Path to save image
        nrow: Number of images per row
        normalize: Whether to normalize images to [0, 1]
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create grid
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, value_range=(-1, 1))
    
    # Convert to PIL image and save
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(output_path)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    elif minutes > 0:
        return f'{minutes}m {secs}s'
    else:
        return f'{secs}s'


def compute_gradient_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients from model as a dictionary.
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Dictionary mapping parameter names to gradient tensors
    """
    gradient_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Clone and detach gradients
            gradient_dict[name] = param.grad.clone().detach().cpu()
    
    return gradient_dict


def apply_gradients(model: torch.nn.Module, gradient_dict: Dict[str, torch.Tensor]):
    """Apply gradients to model parameters.
    
    Args:
        model: PyTorch model
        gradient_dict: Dictionary mapping parameter names to gradients
    """
    for name, param in model.named_parameters():
        if name in gradient_dict:
            if param.grad is None:
                param.grad = gradient_dict[name].to(param.device)
            else:
                param.grad.copy_(gradient_dict[name].to(param.device))


def average_gradients(gradient_dicts: list) -> Dict[str, torch.Tensor]:
    """Average gradients from multiple workers.
    
    Args:
        gradient_dicts: List of gradient dictionaries
        
    Returns:
        Averaged gradient dictionary
    """
    if not gradient_dicts:
        return {}
    
    # Initialize with first gradient dict
    averaged = {}
    for name, grad in gradient_dicts[0].items():
        averaged[name] = grad.clone()
    
    # Sum remaining gradients
    for grad_dict in gradient_dicts[1:]:
        for name, grad in grad_dict.items():
            averaged[name] += grad
    
    # Divide by number of workers
    num_workers = len(gradient_dicts)
    for name in averaged:
        averaged[name] /= num_workers
    
    return averaged


def weighted_average_gradients(gradients_info: list) -> Dict[str, torch.Tensor]:
    """Average gradients with weights based on number of samples.
    
    Args:
        gradients_info: List of dicts with 'gradients' and 'num_samples' keys
        
    Returns:
        Weighted averaged gradient dictionary
    """
    if not gradients_info:
        return {}
    
    total_samples = sum(info['num_samples'] for info in gradients_info)
    
    # Initialize averaged gradients
    averaged = {}
    first_grads = gradients_info[0]['gradients']
    for name in first_grads:
        averaged[name] = torch.zeros_like(first_grads[name])
    
    # Weighted sum
    for info in gradients_info:
        weight = info['num_samples'] / total_samples
        for name, grad in info['gradients'].items():
            averaged[name] += weight * grad
    
    return averaged


def print_training_stats(
    iteration: int,
    epoch: int,
    g_loss: float,
    d_loss: float,
    d_real_acc: float,
    d_fake_acc: float,
    num_workers: int = None
):
    """Print training statistics.
    
    Args:
        iteration: Current iteration
        epoch: Current epoch
        g_loss: Generator loss
        d_loss: Discriminator loss
        d_real_acc: Discriminator accuracy on real images
        d_fake_acc: Discriminator accuracy on fake images
        num_workers: Number of active workers
    """
    stats = f'[Epoch {epoch}] [Iter {iteration}] '
    stats += f'G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f} | '
    stats += f'D_real: {d_real_acc:.2%} | D_fake: {d_fake_acc:.2%}'
    
    if num_workers is not None:
        stats += f' | Workers: {num_workers}'
    
    print(stats)


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    device = get_device()
    print(f'Device: {device}')
    
    gpu_name = get_gpu_name()
    print(f'GPU: {gpu_name}')
    
    hostname = get_hostname()
    print(f'Hostname: {hostname}')
    
    worker_id = generate_worker_id()
    print(f'Worker ID: {worker_id}')
    
    print(format_time(3661))
    print(format_time(125))
    print(format_time(45))
    
    print("\nUtilities tested successfully!")
