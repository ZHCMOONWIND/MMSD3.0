"""
High-performance Data Loader - Based on your provided efficient code structure
"""
import os
import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, RobertaTokenizer
from torchvision import transforms
import warnings

# Filter PIL transparency warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

class FastSarcasmDataset(Dataset):
    """High-performance sarcasm detection dataset"""
    
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.vision_transform = self._initialize_transforms()[mode]
        
        # Initialize tokenizer based on configuration
        if config.text_encoder_type == 'bert':
            from transformers import BertTokenizer
            if config.text_model_variant == 'base':
                tokenizer_name = 'bert-base-uncased'
            else:
                tokenizer_name = 'bert-large-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        else:  # roberta
            from transformers import RobertaTokenizer
            # Use RoBERTa model specified in config
            self.tokenizer = RobertaTokenizer.from_pretrained(config.get_text_model_name())
        
        # Initialize data and OCR data
        self.text_arr, self.img_path, self.label, self.idx2file = self._init_data()
        self.ocr_data = self._load_ocr_data()
        
    def _initialize_transforms(self):
        """Initialize image transforms - using simple and efficient transforms"""
        input_size = 224
        img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        image_transforms = {}
        image_transforms['train'] = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        image_transforms['val'] = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        image_transforms['test'] = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        return image_transforms
    
    def _load_ocr_data(self):
        """Load OCR data"""
        ocr_data = {}
        # MMSD3 uses unified OCR file
        ocr_file = './ocr_results/mmsd3_all_ocr.json'
        
        try:
            print(f"Loading OCR data from {ocr_file}...")
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_list = json.load(f)
            
            # Convert to dictionary format for fast lookup
            for item in ocr_list:
                ocr_data[item['id']] = item['ocr_text']
            
            print(f"Loaded {len(ocr_data)} OCR entries for MMSD3")
            
        except FileNotFoundError:
            print(f"OCR file not found: {ocr_file}")
            print("Will use empty OCR text for all MMSD3 samples")
        except Exception as e:
            print(f"Error loading OCR data: {e}")
        
        return ocr_data
    
    def _init_data(self):
        """Initialize data - based on your efficient implementation"""
        return self._load_mmsd3_data()
    
    def _load_mmsd3_data(self):
        """Load MMSD3 multi-image data"""
        if self.mode == 'train':
            data_path = self.config.mmsd3_train_data_path
        elif self.mode == 'val':
            data_path = self.config.mmsd3_val_data_path
        else:  # test
            data_path = self.config.mmsd3_test_data_path
        
        print(f"Loading MMSD3 data from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_arr, img_path, labels, idx2file = {}, {}, {}, []
        
        # Use data_root from config
        data_root = self.config.data_root
        
        for item in data:
            file_id = str(item['id'])
            text_arr[file_id] = item['text']
            # Labels in MMSD3 dataset are already numbers 0 or 1, use directly
            labels[file_id] = item['label']
            
            # MMSD3 is multi-image, process all images
            if item['images']:
                valid_img_paths = []
                
                for img_path_str in item['images']:
                    # Build full image path - new format: ./images/qlOp17_1.png
                    if img_path_str.startswith('./'):
                        # Remove leading ./
                        relative_path = img_path_str.replace('./', '')
                        full_path = os.path.join(data_root, relative_path)
                    else:
                        # Absolute path or relative path
                        full_path = os.path.join(data_root, img_path_str)
                    
                    if os.path.exists(full_path):
                        valid_img_paths.append(full_path)
                    else:
                        print(f"Image not found: {full_path}")
                
                # Only add sample if at least one image exists
                if valid_img_paths:
                    img_path[file_id] = valid_img_paths  # Store list of all valid image paths
                    idx2file.append(file_id)
        
        print(f"Loaded {len(idx2file)} MMSD3 samples")
        return text_arr, img_path, labels, idx2file
    
    def _get_ocr_text(self, file_name, img_paths):
        """Get corresponding OCR text - return format consistent with images"""
        # MMSD3 needs to process multi-images, each image has corresponding OCR text
        ocr_texts = []
        if isinstance(img_paths, list):
            for img_path in img_paths:
                # New format: ./images/qlOp17_1.png
                # Corresponding OCR ID: qlOp17_1
                base_name = os.path.basename(img_path)  # qlOp17_1.png
                image_id = os.path.splitext(base_name)[0]  # qlOp17_1
                
                # Use image name directly as OCR index
                ocr_text = self.ocr_data.get(image_id, "")
                ocr_texts.append(ocr_text)
            
            return ocr_texts  # Return OCR text list, corresponding to image list
        else:
            # Single image case (shouldn't happen in MMSD3, but for safety)
            return [""]
    
    def __getitem__(self, idx):
        """Get single sample - support multi-image processing and OCR text"""
        try:
            file_name = self.idx2file[idx]
            text = self.text_arr[file_name]
            img_paths = self.img_path[file_name]
            label = self.label[file_name]
            
            # Get OCR text list (corresponding to images)
            ocr_texts = self._get_ocr_text(file_name, img_paths)

            # Process multi-image case (MMSD3)
            images = []
            final_ocr_texts = []  # OCR text corresponding to successfully loaded images
            valid_image_count = 0  # Record actual number of successfully loaded images
            
            for i, img_path in enumerate(img_paths):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = self.vision_transform(img)
                    images.append(img)
                    # Add corresponding OCR text (fill with empty string if OCR text count is insufficient)
                    if i < len(ocr_texts):
                        final_ocr_texts.append(ocr_texts[i])
                    else:
                        final_ocr_texts.append("")
                    valid_image_count += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # For images that fail to load, don't add to list, and don't add corresponding OCR text
                    continue
            
            # Limit image count to max_images
            if len(images) > self.config.max_images:
                images = images[:self.config.max_images]
                final_ocr_texts = final_ocr_texts[:self.config.max_images]
                valid_image_count = self.config.max_images
            
            # If no images loaded successfully, add a zero image and empty OCR
            if len(images) == 0:
                images.append(torch.zeros(3, 224, 224))
                final_ocr_texts.append("")
                valid_image_count = 0  # Mark as 0 valid images
            
            # Stack images as tensor, no padding
            images = torch.stack(images, dim=0)  # [actual_num_images, C, H, W]
            ocr_texts = final_ocr_texts  # Use processed OCR text list
            
            # Return original text, OCR text list (corresponding to images), and actual valid image count, batch tokenize and pad in DataLoader's collate_fn
            return file_name, images, text, ocr_texts, label, valid_image_count
            
        except Exception as e:
            print(f"FATAL ERROR processing sample {idx}: {e}")
            print(f"File name: {self.idx2file[idx] if idx < len(self.idx2file) else 'INDEX_OUT_OF_RANGE'}")
            # Return a default sample to avoid training interruption
            return f'error_{idx}', torch.zeros(1, 3, 224, 224), "Error sample", [""], 0, 0

    def __len__(self):
        return len(self.idx2file)


def fast_collate_fn(batch, tokenizer, max_length=512):
    file_names, images, texts, ocr_texts_list, labels, valid_image_counts = zip(*batch)
    
    # Batch tokenization - original texts
    text_encoding = tokenizer(
        list(texts), 
        padding='longest', 
        truncation=True, 
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Process multi-image case
    # Each sample's images shape: [actual_num_images, C, H, W] (not padded)
    # valid_image_counts: Actual valid image count for each sample
    
    max_num_images = max(img.shape[0] for img in images)  # Find maximum image count
    batch_size = len(images)
    C, H, W = images[0].shape[1:]  # Image channels, height, width
    
    # Create padded image tensor
    padded_images = torch.zeros(batch_size, max_num_images, C, H, W)
    image_masks = torch.zeros(batch_size, max_num_images, dtype=torch.bool)  # Mark valid images
    
    # Prepare OCR texts for batch tokenization
    # Flatten all OCR texts into a list, then reshape
    all_ocr_texts = []
    ocr_text_positions = []  # Record position of each sample's OCR texts in flattened list
    
    for i, (img_tensor, valid_count, ocr_texts) in enumerate(zip(images, valid_image_counts, ocr_texts_list)):
        actual_num_imgs = img_tensor.shape[0]
        # Pad all actual images
        padded_images[i, :actual_num_imgs] = img_tensor
        # Mask only marks truly valid image count
        image_masks[i, :valid_count] = True
        
        # Process OCR texts: ensure OCR text count matches max_num_images
        sample_ocr_texts = []
        for j in range(max_num_images):
            if j < len(ocr_texts):
                sample_ocr_texts.append(ocr_texts[j])
            else:
                sample_ocr_texts.append("")  # Fill with empty string
        
        # Record position and add to flattened list
        start_pos = len(all_ocr_texts)
        all_ocr_texts.extend(sample_ocr_texts)
        ocr_text_positions.append((start_pos, start_pos + max_num_images))
    
    # Batch tokenization - all OCR texts
    ocr_encoding = tokenizer(
        all_ocr_texts, 
        padding='longest', 
        truncation=True, 
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Reshape OCR encoding results
    ocr_input_ids = ocr_encoding['input_ids'].view(batch_size, max_num_images, -1)
    ocr_attention_mask = ocr_encoding['attention_mask'].view(batch_size, max_num_images, -1)
    
    # Convert labels
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'file_names': file_names,
        'text_input_ids': text_encoding['input_ids'],
        'text_attention_mask': text_encoding['attention_mask'],
        'ocr_input_ids': ocr_input_ids,  # [batch_size, max_num_images, seq_len]
        'ocr_attention_mask': ocr_attention_mask,  # [batch_size, max_num_images, seq_len]
        'image_pixel_values': padded_images,  # [batch_size, max_num_images, C, H, W]
        'image_masks': image_masks,  # [batch_size, max_num_images] Mark valid images
        'label': labels
    }


class FastDataLoader:
    def __init__(self, config):
        self.config = config
    
    def _show_statistics(self, train_dataset, val_dataset, test_dataset):
        """Display data statistics"""
        for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
            label_counts = {}
            for file_name in dataset.idx2file:
                label = dataset.label[file_name]
                label_text = 'sarcasm' if label == 1 else 'non-sarcasm'
                label_counts[label_text] = label_counts.get(label_text, 0) + 1
            print(f"{name}: {len(dataset)} samples, distribution: {label_counts}")
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        print(f"Creating fast dataloaders for {self.config.dataset}...")
        
        # Create datasets
        train_dataset = FastSarcasmDataset(self.config, 'train')
        val_dataset = FastSarcasmDataset(self.config, 'val')
        test_dataset = FastSarcasmDataset(self.config, 'test')
        
        # Display data statistics
        self._show_statistics(train_dataset, val_dataset, test_dataset)
        
        # Use training set's tokenizer as common tokenizer
        tokenizer = train_dataset.tokenizer
        
        # Create collate function
        def collate_fn(batch):
            return fast_collate_fn(batch, tokenizer, self.config.max_length)
        
        # Create DataLoader - key performance optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4 if os.name != 'nt' else 0,  # Windows uses 0
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if os.name != 'nt' else None,
            persistent_workers=True if os.name != 'nt' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4 if os.name != 'nt' else 0,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if os.name != 'nt' else None,
            persistent_workers=True if os.name != 'nt' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4 if os.name != 'nt' else 0,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if os.name != 'nt' else None,
            persistent_workers=True if os.name != 'nt' else False
        )
        
        return train_loader, val_loader, test_loader


def get_fast_data_statistics(config):
    print(f"Getting statistics for {config.dataset}...")
    
    train_dataset = FastSarcasmDataset(config, 'train')
    val_dataset = FastSarcasmDataset(config, 'val')
    test_dataset = FastSarcasmDataset(config, 'test')
    
    # Calculate label distribution
    for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        label_counts = {}
        for file_name in dataset.idx2file:
            label = dataset.label[file_name]
            label_text = 'sarcasm' if label == 1 else 'non-sarcasm'
            label_counts[label_text] = label_counts.get(label_text, 0) + 1
        print(f"{name}: {len(dataset)} samples, distribution: {label_counts}")
