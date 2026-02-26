"""
Main Training Script
"""
import os
import sys
import torch
import random
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import MultimodalSarcasmDetector
from trainer import Trainer
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn')
def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Create configuration instance
    config = Config()
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Create necessary directories
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    
    print("Multimodal Sarcasm Detection Experiment")
    print("="*50)
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Num Epochs: {config.num_epochs}")
    print(f"Random Seed: {config.random_seed}")
    
    # 1. Load data
    print(f"\n1. Loading data... (Using dataset: {config.dataset})")
    
    print("Using high-performance data loader...")
    from fast_data_loader import FastDataLoader
    
    data_loader = FastDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # 2. Create model components
    print("\n2. Creating model components...")
    
    # Create encoders
    from text_encoder import TextEncoder
    from vision_encoder import VisionEncoder
    
    text_encoder = TextEncoder(config).to(config.device)
    vision_encoder = VisionEncoder(config).to(config.device)
    
    model = MultimodalSarcasmDetector(config).to(config.device)

    
    # 3. Train model
    print("\n3. Starting training...")
    trainer = Trainer(model, text_encoder, vision_encoder, config, train_loader, val_loader, test_loader, config.device)
    epoch_results = trainer.train()
    
    print("\nExperiment completed!")
    print(f"Best validation F1 score: {trainer.best_val_f1:.4f}")
    if epoch_results:
        final_test_f1 = epoch_results[-1]['test_f1']
        print(f"Final test F1 score: {final_test_f1:.4f}")
    print(f"Detailed results per epoch saved at: {os.path.join(config.results_path, 'epoch_results.csv')}")
    print(f"All results saved at: {config.results_path}")
    
    # Display training summary
    if epoch_results:  
        print("\nTraining Summary:")
        
        print("="*70)
        best_val_epoch = max(epoch_results, key=lambda x: x['val_f1'])
        best_test_epoch = max(epoch_results, key=lambda x: x['test_f1'])
        print(f"Best validation performance: Epoch {best_val_epoch['epoch']}, F1={best_val_epoch['val_f1']:.4f}")
        print(f"Best test performance: Epoch {best_test_epoch['epoch']}, F1={best_test_epoch['test_f1']:.4f}")
        print(f"Val-Test F1 gap (final): {abs(epoch_results[-1]['val_f1'] - epoch_results[-1]['test_f1']):.4f}")

if __name__ == "__main__":
    main()
