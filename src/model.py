import torch
import torch.nn as nn
import torch.nn.functional as F

class SequentialModeling(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution layer for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space matrices A and D
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, d_model] Multi-image feature sequence
            
        Returns:
            output: [batch_size, seq_len, d_model] Processed feature sequence
        """
        batch, seqlen, dim = x.shape
        
        # Residual connection
        residual = x
        x = self.norm(x)
        
        # Input projection and gate separation
        xz = self.in_proj(x)  # [batch, seqlen, d_inner * 2]
        x, z = xz.chunk(2, dim=-1)  # [batch, seqlen, d_inner]
        
        # 1D convolution for local dependencies
        x = x.transpose(1, 2)  # [batch, d_inner, seqlen]
        x = self.conv1d(x)[..., :seqlen]  # Truncate padding
        x = x.transpose(1, 2)  # [batch, seqlen, d_inner]
        x = F.silu(x)
        
        # State space parameter computation
        x_dbl = self.x_proj(x)  # [batch, seqlen, d_state * 2]
        delta, B = x_dbl.chunk(2, dim=-1)  # [batch, seqlen, d_state]
        delta = F.softplus(self.dt_proj(x))  # [batch, seqlen, d_inner]
        
        # State space matrix
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan and gating
        y = self._selective_scan(x, delta, A, B)
        y = y * F.silu(z)
        
        # Output projection and residual connection
        output = self.out_proj(y)
        output = output + residual
        
        return output
    
    def _selective_scan(self, u, delta, A, B):
        """
        Selective scan implementation
        
        Selectively updates state based on input, implementing the core functionality of sequence modeling.
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        outputs = []
        for i in range(seqlen):
            # Get input at current timestep
            u_i = u[:, i, :]  # [batch, d_inner]
            delta_i = delta[:, i, :]  # [batch, d_inner]
            B_i = B[:, i, :].unsqueeze(1)  # [batch, 1, d_state]
            
            # State transition matrix computation
            dA = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))  # [batch, d_inner, d_state]
            dB = delta_i.unsqueeze(-1) * B_i  # [batch, d_inner, d_state]
            
            # State update
            h = h * dA + u_i.unsqueeze(-1) * dB
            
            # Output computation
            y_i = torch.sum(h, dim=-1)  # [batch, d_inner]
            outputs.append(y_i)
        
        # Assemble output sequence
        y = torch.stack(outputs, dim=1)  # [batch, seqlen, d_inner]
        
        # Add skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y

class DualStageBridgeModule(nn.Module):
    """
    Args:
        d_model: Input feature dimension
        d_state: State space dimension, default 16
        d_conv: Convolution kernel size, default 4
        expand: Internal dimension expansion factor, default 2
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        self.text_mamba = SequentialModeling(d_model, d_state, d_conv, expand)
        self.vision_mamba = SequentialModeling(d_model, d_state, d_conv, expand)
        
        # === Pre-input cross-modal attention ===
        self.pre_text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.pre_vision_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # === Post-output cross-modal attention ===
        self.post_text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.post_vision_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # === Fusion gating mechanism ===
        self.pre_text_gate = nn.Linear(d_model, d_model)
        self.pre_vision_gate = nn.Linear(d_model, d_model)
        self.post_text_gate = nn.Linear(d_model, d_model)
        self.post_vision_gate = nn.Linear(d_model, d_model)
        
        # === Layer normalization ===
        self.pre_text_norm = nn.LayerNorm(d_model)
        self.pre_vision_norm = nn.LayerNorm(d_model)
        self.post_text_norm = nn.LayerNorm(d_model)
        self.post_vision_norm = nn.LayerNorm(d_model)
        
    def forward(self, text_embeddings, vision_embeddings):
        """
        Forward pass
        
        Args:
            text_embeddings: [batch_size, text_seq_len, d_model] Text sequence
            vision_embeddings: [batch_size, vision_seq_len, d_model] Image sequence
            
        Returns:
            enhanced_text: [batch_size, text_seq_len, d_model] Enhanced text features
            enhanced_vision: [batch_size, vision_seq_len, d_model] Enhanced image features
        """
        # === Stage 1: Pre-input cross-modal attention ===
        # Text queries vision information (Query from text modality, Key/Value from vision modality)
        text_cross_pre, _ = self.pre_text_cross_attn(
            query=text_embeddings,    # Query from text modality
            key=vision_embeddings,    # Key from vision modality
            value=vision_embeddings   # Value from vision modality
        )
        
        # Vision queries text information (Query from vision modality, Key/Value from text modality)
        vision_cross_pre, _ = self.pre_vision_cross_attn(
            query=vision_embeddings,  # Query from vision modality
            key=text_embeddings,      # Key from text modality
            value=text_embeddings     # Value from text modality
        )
        
        # Gating fusion: determine the degree of cross-modal information fusion
        # Use self-features to generate gates, deciding how much cross-modal information is needed
        text_gate_pre = torch.sigmoid(self.pre_text_gate(text_embeddings))  # Generate gate from self-features
        vision_gate_pre = torch.sigmoid(self.pre_vision_gate(vision_embeddings))  # Generate gate from self-features
        
        text_input = self.pre_text_norm(text_embeddings + text_gate_pre * text_cross_pre)
        vision_input = self.pre_vision_norm(vision_embeddings + vision_gate_pre * vision_cross_pre)
        
        # === Stage 2: Mamba Block independent sequence modeling ===
        text_mamba_out = self.text_mamba(text_input)
        vision_mamba_out = self.vision_mamba(vision_input)
        
        # === Stage 3: Post-output cross-modal attention ===
        text_cross_post, _ = self.post_text_cross_attn(
            query=text_mamba_out,
            key=vision_mamba_out,
            value=vision_mamba_out
        )
        
        # Vision Mamba output queries text Mamba output
        vision_cross_post, _ = self.post_vision_cross_attn(
            query=vision_mamba_out,
            key=text_mamba_out,
            value=text_mamba_out
        )
        
        # Gating fusion: final decision on degree of cross-modal enhancement
        text_gate_post = torch.sigmoid(self.post_text_gate(text_mamba_out))
        vision_gate_post = torch.sigmoid(self.post_vision_gate(vision_mamba_out))

        enhanced_text = self.post_text_norm(text_mamba_out + text_gate_post * text_cross_post)
        enhanced_vision = self.post_vision_norm(vision_mamba_out + vision_gate_post * vision_cross_post)
        
        return enhanced_text, enhanced_vision

class MultimodalSarcasmDetector(nn.Module):
    
    def __init__(self, config):
        super(MultimodalSarcasmDetector, self).__init__()
        self.config = config
        self.fusion_dim = config.fusion_dim
        self.embedding_dim = config.embedding_dim
        
        # Initialize model components
        self._init_feature_processors()
        self._init_attention_modules()
        self._init_fusion_modules()
        self._init_classifier()
        
        # Weight initialization
        self._init_all_weights()
    
    def _init_feature_processors(self):
        """Initialize feature processing modules"""
        # Feature processing layers for each modality
        self.text_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.vision_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.ocr_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(0.1),
            nn.GELU()
        )
        
        # Text attention weight calculation
        self.text_attention = nn.Linear(self.embedding_dim, 1)
        
        # Text-image relevance calculation network
        self.text_image_relevance = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 1)
        )
        
        # Relevance calculation projection layers
        self.text_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.image_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Position encoding (for multi-image)
        self.image_position_embedding = nn.Embedding(10, self.embedding_dim)
        
        # Star rating encoding
        self.rating_embedding = nn.Linear(1, self.fusion_dim // 4)
    
    def _init_attention_modules(self):
        """Initialize attention modules"""
        # Four cross-attention modules
        attention_config = {
            'embed_dim': self.embedding_dim,
            'num_heads': 8,
            'dropout': 0.1,
            'batch_first': True
        }
        
        self.vt_cross = nn.MultiheadAttention(**attention_config)  # Text queries vision
        self.tv_cross = nn.MultiheadAttention(**attention_config)  # Vision queries text  
        self.ot_cross = nn.MultiheadAttention(**attention_config)  # Text queries OCR
        self.to_cross = nn.MultiheadAttention(**attention_config)  # OCR queries text
        
        self.multi_image_mamba = SequentialModeling(d_model=self.embedding_dim)
        self.text_sequence_mamba = SequentialModeling(d_model=self.embedding_dim)
        
        # Cross-modal Mamba block
        self.cross_modal_mamba = DualStageBridgeModule(d_model=self.embedding_dim)
    
    def _init_fusion_modules(self):
        """Initialize fusion modules"""
        # Cross-modal feature projection layer - using gated fusion
        self.cross_modal_projection = nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        self.cross_modal_gate = nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        
        # Feature mapping layers (embedding_dim -> fusion_dim)
        self.text_feature_mapper = nn.Linear(self.embedding_dim, self.fusion_dim)
        self.mamba_feature_mapper = nn.Linear(self.embedding_dim, self.fusion_dim)
        self.cross_modal_feature_mapper = nn.Linear(self.embedding_dim, self.fusion_dim)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim * 3 + self.fusion_dim // 4, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
    
    def _init_classifier(self):
        """Initialize classifier"""
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 4, 2)
        )
    
    def _init_all_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(keyword in name for keyword in ['text_fc', 'vision_fc', 'ocr_fc', 
                                                       'text_projection', 'image_projection',
                                                       'text_image_relevance', 'rating_embedding']):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif 'classifier' in name:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        final_layer = self.classifier[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.02)
        if final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, 0.0)
    
    def forward(self, text_embeddings, vision_embeddings, ocr_embeddings, star_rating=None, num_images=None):
        batch_size = text_embeddings.size(0)
        text_seq_len = text_embeddings.size(1)
        
        # 1. Process input feature dimensions and normalize
        vision_embeddings, max_images, num_images = self._process_vision_embeddings(
            vision_embeddings, batch_size, num_images
        )
        ocr_embeddings = self._process_ocr_embeddings(ocr_embeddings)
        
        # Apply feature processing layers
        text_embeddings = self.text_fc(text_embeddings)
        vision_embeddings = self.vision_fc(vision_embeddings)
        ocr_embeddings = self.ocr_fc(ocr_embeddings)
        
        # 2. Add position encoding and create mask
        vision_embeddings = self._add_position_encoding(vision_embeddings, max_images)
        mask = self._create_mask(num_images, max_images, vision_embeddings.device)
        
        # 3. Cross-modal sequence modeling
        text_mamba_output, vision_mamba_output = self.cross_modal_mamba(
            text_embeddings, vision_embeddings
        )
        text_aggregated = torch.mean(text_mamba_output, dim=1)  # [batch_size, embedding_dim]
        vision_aggregated = torch.mean(vision_mamba_output, dim=1)  # [batch_size, embedding_dim]

        # 4. Compute text-image relevance weights (using aggregated text sequence representation)
        text_cls = torch.mean(text_embeddings, dim=1)
        relevance_weights = self._compute_relevance_weights(
            text_cls, vision_embeddings, mask, max_images
        )
        
        # 5. Modal cross-attention fusion
        cross_modal_features = self._Relevance_Guided_Fusion(
            text_embeddings, vision_embeddings, ocr_embeddings, relevance_weights
        )
        
        
        # 7. Feature mapping and final fusion
        final_features = self._final_feature_fusion(
            text_aggregated, vision_aggregated, cross_modal_features, star_rating
        )
        
        # 8. Classification
        logits = self.classifier(final_features)
        return logits
    
    def _process_vision_embeddings(self, vision_embeddings, batch_size, num_images):
        """Process vision feature dimensions"""
        # Multi-image case: [batch_size, max_images, embedding_dim]
        max_images = vision_embeddings.size(1)
        if num_images is None:
            num_images = torch.full((batch_size,), max_images, dtype=torch.long, device=vision_embeddings.device)
        
        return vision_embeddings, max_images, num_images
    
    def _process_ocr_embeddings(self, ocr_embeddings):
        """Process OCR feature dimensions"""
        # Multi-image case: [batch_size, max_images, embedding_dim]
        return ocr_embeddings
    
    def _add_position_encoding(self, vision_embeddings, max_images):
        """Add position encoding for multi-image"""
        if max_images > 1:
            batch_size = vision_embeddings.size(0)
            for i in range(max_images):
                position_ids = torch.full((batch_size,), i, dtype=torch.long, device=vision_embeddings.device)
                position_embedding = self.image_position_embedding(position_ids)
                vision_embeddings[:, i, :] = vision_embeddings[:, i, :] + position_embedding
        return vision_embeddings
    
    def _create_mask(self, num_images, max_images, device):
        """Create mask to ignore padded images"""
        mask = None
        if num_images is not None and max_images > 1:
            mask = torch.arange(max_images, device=device).unsqueeze(0) < num_images.unsqueeze(1)
            mask = mask.float()
        return mask
    
    def _compute_relevance_weights(self, text_embeddings, vision_embeddings, mask, max_images):
        """Compute text-image relevance weights"""
        # Cosine similarity method
        text_projected = self.text_projection(text_embeddings)
        text_projected = F.normalize(text_projected, p=2, dim=-1)
        
        vision_projected = self.image_projection(vision_embeddings)
        vision_projected = F.normalize(vision_projected, p=2, dim=-1)
        
        cosine_similarity = torch.bmm(
            vision_projected, 
            text_projected.unsqueeze(-1)
        ).squeeze(-1)
        
        # Learnable relevance network
        expanded_text = text_embeddings.unsqueeze(1).expand(-1, max_images, -1)
        text_image_pairs = torch.cat([expanded_text, vision_embeddings], dim=-1)
        learned_relevance = self.text_image_relevance(text_image_pairs).squeeze(-1)
        
        # Combine two relevance calculation methods
        alpha = 0.6
        combined_relevance = alpha * cosine_similarity + (1 - alpha) * learned_relevance
        #combined_relevance = cosine_similarity
        # Apply mask
        if mask is not None:
            combined_relevance = combined_relevance.masked_fill(~mask.bool(), float('-inf'))
        
        relevance_weights = F.softmax(combined_relevance, dim=-1)
        return relevance_weights

    def _Relevance_Guided_Fusion(self, text_embeddings, vision_embeddings, ocr_embeddings, relevance_weights):
        """Fine-grained cross-modal fusion based on relevance weights"""
        batch_size, text_seq_len, _ = text_embeddings.shape
        _, max_images, _ = vision_embeddings.shape
        
        # === Core design: Both text and vision interact with OCR ===
        
        # 1. Text-OCR interaction (text understands visual content through OCR)
        text_attend_ocr, _ = self.ot_cross(
            query=text_embeddings,  # [batch_size, text_seq_len, embedding_dim]
            key=ocr_embeddings,     # [batch_size, max_images, embedding_dim]
            value=ocr_embeddings
        )
        # Residual connection
        text_attend_ocr = text_attend_ocr + text_embeddings  # [batch_size, text_seq_len, embedding_dim]

        # 2. Vision-OCR interaction (vision aligns with text through OCR)
        vision_attend_ocr, _ = self.to_cross(
            query=vision_embeddings,  # [batch_size, max_images, embedding_dim]
            key=ocr_embeddings,       # [batch_size, max_images, embedding_dim]
            value=ocr_embeddings
        )
        # Residual connection
        vision_attend_ocr = vision_attend_ocr + vision_embeddings  # [batch_size, max_images, embedding_dim]

        
        # 3. Aggregate text features
        text_aggregated = torch.mean(text_attend_ocr, dim=1)  # [batch_size, embedding_dim]
        
        # 4. Per-image text-vision interaction, modulated by relevance weights
        fused_features = torch.zeros_like(text_aggregated)  # [batch_size, embedding_dim]
        
        for i in range(max_images):
            vision_i = vision_attend_ocr[:, i, :]  # [batch_size, embedding_dim] - i-th image feature
            weight_i = relevance_weights[:, i].unsqueeze(-1)  # [batch_size, 1] - relevance weight for i-th image
            
            # Weighted interaction between text and i-th image
            interaction_i = text_aggregated * vision_i * weight_i  # [batch_size, embedding_dim]
            fused_features = fused_features + interaction_i
        
        return fused_features  # [batch_size, embedding_dim]

    def _final_feature_fusion(self, weighted_text, mamba_weighted_output, cross_modal_features, star_rating):
        """Final feature fusion"""
        batch_size = weighted_text.size(0)
        
        # Map features to fusion_dim
        text_fusion_features = self.text_feature_mapper(weighted_text)
        mamba_fusion_features = self.mamba_feature_mapper(mamba_weighted_output)
        cross_modal_fusion_features = self.cross_modal_feature_mapper(cross_modal_features)
        
        # Concatenate features
        concat_features = torch.cat([
            text_fusion_features,  
            mamba_fusion_features, 
            cross_modal_fusion_features
        ], dim=1)
        
        # Add star rating information
        if star_rating is not None:
            rating_features = self.rating_embedding(star_rating.unsqueeze(-1))
            concat_features = torch.cat([concat_features, rating_features], dim=1)
        else:
            zero_rating = torch.zeros(batch_size, self.fusion_dim // 4, device=concat_features.device)
            concat_features = torch.cat([concat_features, zero_rating], dim=1)
        
        # Final fusion
        fused_features = self.fusion_layer(concat_features)
        return fused_features
