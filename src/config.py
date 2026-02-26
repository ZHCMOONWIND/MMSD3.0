"""
Configuration file: Define experiment hyperparameters and paths
"""
import os
from dataclasses import dataclass, field

# Lazy import torch to avoid circular dependencies
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

@dataclass
class Config:
    # Dataset selection: 'MMSD3'
    dataset: str = "MMSD3"
    use_fast_loader: bool = True  # Use high-performance data loader
    
    # Data root directory
    data_root: str = "./data"  # Root directory of the dataset
    
    # Data paths - automatically set based on dataset type
    train_data_path: str = field(init=False)
    val_data_path: str = field(init=False)
    test_data_path: str = field(init=False)
    image_path: str = field(init=False)
    
    # Multi-image data paths for MMSD3
    mmsd3_train_data_path: str = "./data/MMSD3/train_data.json"
    mmsd3_val_data_path: str = "./data/MMSD3/val_data.json"
    mmsd3_test_data_path: str = "./data/MMSD3/test_data.json"
    mmsd3_image_path: str = "./reorganized_images"
    
    # Encoder selection
    text_encoder_type: str = "roberta"  # 'bert' or 'roberta'
    vision_encoder_type: str = "vit"   # 'vit' or 'clip'
    text_model_variant: str = "base"  # 'base' or 'large'
    vision_model_variant: str = "base"  # 'base' or 'large'
    
    # Model parameters - selected based on encoder type
    bert_model_name: str = "bert-base-uncased"
    roberta_model_name: str = "cardiffnlp/twitter-roberta-base-emoji"
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 20
    max_length: int = 512
    max_images: int = 4  # Maximum number of images per sample
    

    vision_lr: float = 2e-5
    vision_weight_decay: float = 1e-5
    text_lr: float = 2e-5
    text_weight_decay: float = 1e-5
    multimodal_lr: float = 2e-5
    multimodal_weight_decay: float = 1e-5
    
    freeze_encoders: bool = False  # Whether to freeze encoder parameters
    freeze_vision_layers: int = 4  # Freeze first N layers of vision encoder (0 = all trainable)
    freeze_text_layers: int = 4    # Freeze first N layers of text encoder (0 = all trainable)
    
    # Feature dimensions - automatically set based on encoder type
    text_feature_dim: int = field(init=False)  # BERT/RoBERTa hidden size
    vision_feature_dim: int = field(init=False)  # ViT/CLIP vision feature size
    embedding_dim: int = 768  # Unified encoder output dimension
    fusion_dim: int = 512  # Fused feature dimension
    
    # Device settings
    device: str = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    # Output paths
    model_save_path: str = "./models"
    results_path: str = "./results"
    
    # Random seed
    random_seed: int = 2026
    
    def __post_init__(self):
        """Set configuration based on dataset type and encoder type"""
        # Set data paths
        if self.dataset == "MMSD3":
            self.train_data_path = self.mmsd3_train_data_path
            self.val_data_path = self.mmsd3_val_data_path
            self.test_data_path = self.mmsd3_test_data_path
            self.image_path = self.mmsd3_image_path
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset}. Supported types: MMSD3")
        
        # Set feature dimensions
        if self.text_encoder_type == "bert":
            self.text_feature_dim = 768  # BERT-base hidden size
        elif self.text_encoder_type == "roberta":
            self.text_feature_dim = 768  # RoBERTa-base hidden size
        else:
            raise ValueError(f"Unsupported text encoder type: {self.text_encoder_type}. Supported types: bert, roberta")
        
        if self.vision_encoder_type == "vit":
            self.vision_feature_dim = 768  # ViT-base hidden size
        elif self.vision_encoder_type == "clip":
            self.vision_feature_dim = 512  # CLIP vision feature size
        else:
            raise ValueError(f"Unsupported vision encoder type: {self.vision_encoder_type}. Supported types: vit, clip")
    
    def get_text_model_name(self):
        """Get text model name"""
        if self.text_encoder_type == "bert":
            return self.bert_model_name
        elif self.text_encoder_type == "roberta":
            return self.roberta_model_name
        else:
            raise ValueError(f"Unsupported text encoder type: {self.text_encoder_type}")
    
    def get_vision_model_name(self):
        """Get vision model name"""
        if self.vision_encoder_type == "vit":
            return self.vit_model_name
        elif self.vision_encoder_type == "clip":
            return self.clip_model_name
        else:
            raise ValueError(f"Unsupported vision encoder type: {self.vision_encoder_type}")
    
    def is_multimodal_dataset(self):
        """Check if it's a multi-image dataset"""
        return self.dataset == "MMSD3"

# Set device after class definition
def get_device():
    """Get computing device"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# Update Config.device
if _TORCH_AVAILABLE:
    Config.device = "cuda" if torch.cuda.is_available() else "cpu"
