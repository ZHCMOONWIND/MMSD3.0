"""
Vision Encoder: Image feature extraction supporting ViT and CLIP
"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, ViTModel

class VisionEncoder(nn.Module):
    """Vision encoder supporting ViT and CLIP"""
    
    def __init__(self, config):
        super(VisionEncoder, self).__init__()
        self.config = config
        self.encoder_type = config.vision_encoder_type
        
        # Load the pretrained model based on configuration
        if self.encoder_type == "clip":
            # Load pretrained CLIP vision model - ignore irrelevant warnings
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
                self.vision_model = CLIPVisionModel.from_pretrained(config.get_vision_model_name())
            print(f"Loading CLIP vision model: {config.get_vision_model_name()}")
        elif self.encoder_type == "vit":
            self.vision_model = ViTModel.from_pretrained(config.get_vision_model_name())
            print(f"Loading ViT model: {config.get_vision_model_name()}")
        else:
            raise ValueError(f"Unsupported vision encoder type: {self.encoder_type}")
        
        # Optional: freeze encoder parameters
        if hasattr(config, 'freeze_encoders') and config.freeze_encoders:
            self._freeze_all_layers()
        elif hasattr(config, 'freeze_vision_layers'):
            # Use custom number of frozen layers
            self._freeze_layers(config.freeze_vision_layers)
        else:
            # Default: freeze first 4 layers
            self._freeze_layers(8)
        
        # Feature projection layer - output to unified embedding_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(self.vision_model.config.hidden_size, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Image count encoding (for handling multi-image scenarios)
        self.image_count_embedding = nn.Embedding(10, config.embedding_dim)  # Assume max 10 images
        
    def _freeze_layers(self, freeze_layers=0):
        """Freeze the first few layers' parameters"""
        if self.encoder_type == "clip":
            layers = self.vision_model.vision_model.encoder.layers
        elif self.encoder_type == "vit":
            layers = self.vision_model.encoder.layer
        
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def _freeze_all_layers(self):
        """Freeze all parameters of the vision encoder"""
        print(f"Freezing all parameters of {self.encoder_type.upper()} vision encoder...")
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values):
        """
        Forward pass
        Args:
            pixel_values: [batch_size, channels, height, width] Single image input
        Returns:
            vision_features: [batch_size, embedding_dim]
        """
        # Vision encoding
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
        
        # Get pooled output
        if self.encoder_type == "clip":
            pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        elif self.encoder_type == "vit":
            # ViT uses [CLS] token representation
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None else vision_outputs.last_hidden_state[:, 0, :]
        
        # Feature projection
        vision_features = self.feature_projection(pooled_output)  # [batch_size, fusion_dim]
        
        return vision_features
    
    def get_attention_maps(self, pixel_values):
        """Get attention maps for visualization"""
        with torch.no_grad():
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True
            )
            return outputs.attentions
