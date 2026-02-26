"""
Text Encoder: Text feature extraction supporting BERT and RoBERTa
"""
import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class TextEncoder(nn.Module):
    """Text encoder supporting BERT and RoBERTa"""
    
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.encoder_type = config.text_encoder_type
        
        # Load the pretrained model based on configuration
        if self.encoder_type == "bert":
            self.text_model = BertModel.from_pretrained(config.get_text_model_name())
            print(f"Loading BERT model: {config.get_text_model_name()}")
        elif self.encoder_type == "roberta":
            self.text_model = RobertaModel.from_pretrained(config.get_text_model_name())
            print(f"Loading RoBERTa model: {config.get_text_model_name()}")
        else:
            raise ValueError(f"Unsupported text encoder type: {self.encoder_type}")
        
        # Optional: freeze encoder parameters
        if hasattr(config, 'freeze_encoders') and config.freeze_encoders:
            self._freeze_all_layers()
        elif hasattr(config, 'freeze_text_layers'):
            # Use custom number of frozen layers
            self._freeze_layers(config.freeze_text_layers)
        else:
            # Default: do not freeze any layers (all trainable)
            self._freeze_layers(8)
        
        # Feature projection layer - output to unified embedding_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _freeze_layers(self, freeze_layers=0):
        """Freeze the first few layers' parameters"""
        if self.encoder_type == "bert":
            layers = self.text_model.encoder.layer
        elif self.encoder_type == "roberta":
            layers = self.text_model.encoder.layer
        
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def _freeze_all_layers(self):
        """Freeze all parameters of the text encoder"""
        print(f"Freezing all parameters of {self.encoder_type.upper()} text encoder...")
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, return_sequence=False):
        """
        Forward pass
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_sequence: bool, whether to return full sequence
        Returns:
            If return_sequence=True: [batch_size, seq_len, embedding_dim]
            If return_sequence=False: [batch_size, embedding_dim] (CLS token)
        """
        # Text encoding
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if return_sequence:
            # Return full sequence representation
            sequence_output = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            # Project features at each position
            text_features = self.feature_projection(sequence_output)  # [batch_size, seq_len, embedding_dim]
        else:
            # Return CLS token representation
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                pooled_output = text_outputs.pooler_output
            else:
                # If no pooler_output, use the hidden state at [CLS] position
                pooled_output = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
            # Feature projection
            text_features = self.feature_projection(pooled_output)  # [batch_size, embedding_dim]
        
        return text_features
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Get attention weights for visualization"""
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            return outputs.attentions
