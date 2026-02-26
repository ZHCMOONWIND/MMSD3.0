"""
Training Module
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import warnings

# Filter PIL transparency warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

class Trainer:
    """Model trainer"""
    
    def __init__(self, model, text_encoder, vision_encoder, config, train_loader, val_loader, test_loader, device):
        self.model = model
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Calculate class weights to handle data imbalance
        class_counts = [0, 0]  # [non-sarcasm, sarcasm]
        for batch in train_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label.item()] += 1
        
        # Calculate weights: less frequent classes get higher weights
        total_samples = sum(class_counts)
        # Add safety check to avoid division by zero
        class_weights = []
        for count in class_counts:
            if count == 0:
                print("Warning: A class has 0 samples, unable to calculate weights!!!!")
                # If a class has 0 samples, give it a very small weight
                class_weights.append(0.001)
            else:
                class_weights.append(total_samples / (2 * count))
        class_weights = torch.tensor(class_weights, device=device)
        
        print(f"Class distribution: non-sarcasm={class_counts[0]}, sarcasm={class_counts[1]}")
        print(f"Class weights: non-sarcasm={class_weights[0]:.3f}, sarcasm={class_weights[1]:.3f}")
        
        # Loss function - using weighted cross entropy
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create separate optimizers and schedulers for different components
        self._setup_optimizers_and_schedulers(config)
        
        # Record training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'test_loss': [],
            'test_acc': [],
            'test_f1': []
        }
        
        # Detailed results per epoch
        self.epoch_results = []
        
        # Best model saving
        self.best_val_f1 = 0.0
    
    def _setup_optimizers_and_schedulers(self, config):
        """Set up separate optimizers and schedulers"""
        # Vision encoder optimizer
        self.vision_optimizer = AdamW(
            self.vision_encoder.parameters(),
            lr=config.vision_lr,
            weight_decay=config.vision_weight_decay
        )
        
        # Text encoder optimizer  
        self.text_optimizer = AdamW(
            self.text_encoder.parameters(),
            lr=config.text_lr,
            weight_decay=config.text_weight_decay
        )
        
        # Multimodal fusion module optimizer
        self.multimodal_optimizer = AdamW(
            self.model.parameters(),
            lr=config.multimodal_lr,
            weight_decay=config.multimodal_weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * config.num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        # Vision encoder scheduler
        self.vision_scheduler = get_linear_schedule_with_warmup(
            self.vision_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Text encoder scheduler
        self.text_scheduler = get_linear_schedule_with_warmup(
            self.text_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Multimodal scheduler
        self.multimodal_scheduler = get_linear_schedule_with_warmup(
            self.multimodal_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def train_epoch(self):
        """Train one epoch"""
        self.text_encoder.train()
        self.vision_encoder.train()
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            try:
                # Text encoding - return full sequence for model
                text_features = self.text_encoder(
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch['text_attention_mask'],
                    return_sequence=True  # Return full sequence
                )
                
                # Process OCR text encoding
                ocr_input_ids = batch['ocr_input_ids']
                ocr_attention_mask = batch['ocr_attention_mask']
                
                # OCR text encoding - Multi-image case (MMSD3) - OCR is [batch_size, max_images, seq_len]
                batch_size, max_images, seq_len = ocr_input_ids.shape
                ocr_features_list = []
                
                for i in range(max_images):
                    single_ocr_ids = ocr_input_ids[:, i, :]  # [batch_size, seq_len]
                    single_ocr_mask = ocr_attention_mask[:, i, :]  # [batch_size, seq_len]
                    
                    single_ocr_features = self.text_encoder(
                        input_ids=single_ocr_ids,
                        attention_mask=single_ocr_mask,
                        return_sequence=False  # Return CLS token
                    )  # [batch_size, embedding_dim]
                    ocr_features_list.append(single_ocr_features)
                
                ocr_features = torch.stack(ocr_features_list, dim=1)  # [batch_size, max_images, embedding_dim]
                
                # Process image encoding - Multi-image case (MMSD3)
                image_pixel_values = batch['image_pixel_values']
                image_masks = batch.get('image_masks', None)  # Get image masks
                
                batch_size, max_images = image_pixel_values.shape[:2]
                image_features_list = []
                
                for i in range(max_images):
                    single_image = image_pixel_values[:, i, :, :, :]
                    single_image_features = self.vision_encoder(pixel_values=single_image)
                    image_features_list.append(single_image_features)
                
                vision_features = torch.stack(image_features_list, dim=1)
                
                # If there are image masks, calculate actual image count
                if image_masks is not None:
                    num_images = image_masks.sum(dim=1)  # [batch_size] Actual image count per sample
                else:
                    num_images = torch.full((batch_size,), max_images, dtype=torch.long, device=self.device)
                
                # Multimodal fusion and classification
                logits = self.model(text_features, vision_features, ocr_features,
                                  star_rating=batch.get('star_rating', None),
                                  num_images=num_images)
                loss = self.criterion(logits, batch['label'])
            except Exception as e:
                print(f"Forward pass error: {e}")
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()]}")
                raise e
            
            # Backward pass
            self.vision_optimizer.zero_grad()
            self.text_optimizer.zero_grad()
            self.multimodal_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.vision_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            self.vision_optimizer.step()
            self.text_optimizer.step()
            self.multimodal_optimizer.step()
            
            # Update learning rates
            self.vision_scheduler.step()
            self.text_scheduler.step()
            self.multimodal_scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch['label']).sum().item()
            total_samples += batch['label'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def evaluate_dataset(self, data_loader, dataset_name):
        """Evaluate specified dataset"""
        self.model.eval()
        self.text_encoder.eval()
        self.vision_encoder.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                try:
                    # Text encoding - return full sequence for model
                    text_features = self.text_encoder(
                        input_ids=batch['text_input_ids'],
                        attention_mask=batch['text_attention_mask'],
                        return_sequence=True  # Return full sequence
                    )
                    
                    # Process OCR text encoding
                    ocr_input_ids = batch['ocr_input_ids']
                    ocr_attention_mask = batch['ocr_attention_mask']
                    
                    # OCR text encoding - Multi-image case (MMSD3) - OCR is [batch_size, max_images, seq_len]
                    batch_size, max_images, seq_len = ocr_input_ids.shape
                    ocr_features_list = []
                    
                    for i in range(max_images):
                        single_ocr_ids = ocr_input_ids[:, i, :]  # [batch_size, seq_len]
                        single_ocr_mask = ocr_attention_mask[:, i, :]  # [batch_size, seq_len]
                        
                        single_ocr_features = self.text_encoder(
                            input_ids=single_ocr_ids,
                            attention_mask=single_ocr_mask,
                            return_sequence=False  # Return CLS token
                        )  # [batch_size, embedding_dim]
                        ocr_features_list.append(single_ocr_features)
                    
                    ocr_features = torch.stack(ocr_features_list, dim=1)  # [batch_size, max_images, embedding_dim]
                    
                    # Process image encoding - Multi-image case (MMSD3)
                    image_pixel_values = batch['image_pixel_values']
                    image_masks = batch.get('image_masks', None)  # Get image masks
                    
                    batch_size, max_images = image_pixel_values.shape[:2]
                    image_features_list = []
                    
                    for i in range(max_images):
                        single_image = image_pixel_values[:, i, :, :, :]
                        single_image_features = self.vision_encoder(pixel_values=single_image)
                        image_features_list.append(single_image_features)
                    
                    vision_features = torch.stack(image_features_list, dim=1)
                    
                    # If there are image masks, calculate actual image count
                    if image_masks is not None:
                        num_images = image_masks.sum(dim=1)  # [batch_size] Actual image count per sample
                    else:
                        num_images = torch.full((batch_size,), max_images, dtype=torch.long, device=self.device)
                    
                    # Multimodal fusion and classification
                    logits = self.model(text_features, vision_features, ocr_features,
                                      star_rating=batch.get('star_rating', None),
                                      num_images=num_images)
                    loss = self.criterion(logits, batch['label'])
                except Exception as e:
                    print(f"Forward pass error during evaluation: {e}")
                    print(f"Batch keys: {list(batch.keys())}")
                    print(f"Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()]}")
                    raise e
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Binary metrics (for binary classification)
        binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        # Macro metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
            'binary_f1': binary_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self):
        """Complete training process"""
        print("Starting training...")
        print(f"Training device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_results = self.evaluate_dataset(self.val_loader, "Validation")
            
            # Testing
            test_results = self.evaluate_dataset(self.test_loader, "Test")
            
            # Record history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_results['loss'])
            self.train_history['val_acc'].append(val_results['accuracy'])
            self.train_history['val_f1'].append(val_results['binary_f1'])  # Use binary_f1 as main F1 metric
            self.train_history['test_loss'].append(test_results['loss'])
            self.train_history['test_acc'].append(test_results['accuracy'])
            self.train_history['test_f1'].append(test_results['binary_f1'])  # Use binary_f1 as main F1 metric
            
            # Record detailed results per epoch
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'val_binary_precision': val_results['binary_precision'],
                'val_binary_recall': val_results['binary_recall'],
                'val_binary_f1': val_results['binary_f1'],
                'val_macro_precision': val_results['macro_precision'],
                'val_macro_recall': val_results['macro_recall'],
                'val_macro_f1': val_results['macro_f1'],
                'test_loss': test_results['loss'],
                'test_acc': test_results['accuracy'],
                'test_binary_precision': test_results['binary_precision'],
                'test_binary_recall': test_results['binary_recall'],
                'test_binary_f1': test_results['binary_f1'],
                'test_macro_precision': test_results['macro_precision'],
                'test_macro_recall': test_results['macro_recall'],
                'test_macro_f1': test_results['macro_f1']
            }
            self.epoch_results.append(epoch_result)
            
            # Print results
            print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
            print(f"Val loss: {val_results['loss']:.4f}, Val accuracy: {val_results['accuracy']:.4f}")
            print(f"Val Binary - F1: {val_results['binary_f1']:.4f}, Precision: {val_results['binary_precision']:.4f}, Recall: {val_results['binary_recall']:.4f}")
            print(f"Val Macro - F1: {val_results['macro_f1']:.4f}, Precision: {val_results['macro_precision']:.4f}, Recall: {val_results['macro_recall']:.4f}")
            print(f"Test loss: {test_results['loss']:.4f}, Test accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Binary - F1: {test_results['binary_f1']:.4f}, Precision: {test_results['binary_precision']:.4f}, Recall: {test_results['binary_recall']:.4f}")
            print(f"Test Macro - F1: {test_results['macro_f1']:.4f}, Precision: {test_results['macro_precision']:.4f}, Recall: {test_results['macro_recall']:.4f}")
            
            # Save best model (using binary_f1 as main metric)
            if val_results['binary_f1'] > self.best_val_f1:
                self.best_val_f1 = val_results['binary_f1']
                model_filename = f"{self.config.dataset.lower()}_best_model_f1_{val_results['binary_f1']:.4f}.pt"
                self.save_model(model_filename)
                print(f"Saving new best model (Val Binary F1: {val_results['binary_f1']:.4f})")
            
            # Save results to CSV per epoch
            self.save_epoch_results()
        
        print("\nTraining completed!")
        print(f"Best validation Binary F1 score: {self.best_val_f1:.4f}")
        
        # Save final training history
        self.save_training_history()
        
        return self.epoch_results
    
    def save_epoch_results(self):
        """Save results per epoch to CSV file"""
        os.makedirs(self.config.results_path, exist_ok=True)
        csv_filename = f'epoch_results_{self.config.dataset.lower()}.csv'
        csv_path = os.path.join(self.config.results_path, csv_filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.epoch_results)
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Epoch results saved to: {csv_path}")
    
    def save_model(self, filename):
        """Save model"""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        save_path = os.path.join(self.config.model_save_path, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'vision_encoder_state_dict': self.vision_encoder.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    def save_training_history(self):
        """Save training history"""
        os.makedirs(self.config.results_path, exist_ok=True)
        history_filename = f'training_history_{self.config.dataset.lower()}.json'
        history_path = os.path.join(self.config.results_path, history_filename)
        
        # Convert numpy types to native Python types
        history_to_save = {}
        for key, values in self.train_history.items():
            history_to_save[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in values]
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Training history saved to: {history_path}")
    
    def load_model(self, model_path):
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load encoder states (if they exist)
        if 'text_encoder_state_dict' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        if 'vision_encoder_state_dict' in checkpoint:
            self.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'])
            
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        print(f"Model loaded from {model_path}")
