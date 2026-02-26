#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Text Extraction Script
Use PaddleOCR to extract OCR text from images in MMSD1, MMSD2, and MMSD3 datasets
"""

import os
import json
import ast
from pathlib import Path
from tqdm import tqdm
import logging
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Union
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRExtractor:
    def __init__(self, use_gpu=True, lang='ch'):
        """
        Initialize OCR extractor
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            lang (str): Recognition language, 'ch' for Chinese-English mixed, 'en' for English
        """
        logger.info("Initializing PaddleOCR...")
        
        # Initialize PaddleOCR (according to 3.x version official documentation)
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,  # Disable document orientation classification
            use_doc_unwarping=False,  # Disable document unwarping
            use_textline_orientation=False,  # Disable text line orientation classifier
            lang=lang
        )
        logger.info(f"PaddleOCR initialization completed, language: {lang}")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract OCR text from a single image
        
        Args:
            image_path (str): Image path
            
        Returns:
            str: Extracted text, multiple lines joined with spaces
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file does not exist: {image_path}")
                return ""
            
            # Perform OCR recognition (using 3.x version predict method)
            result = self.ocr.predict(input=image_path)
            
            # Extract text (according to 3.x version actual return format)
            texts = []
            if result and isinstance(result, list) and len(result) > 0:
                # Get the first result item
                result_item = result[0]
                
                # Extract recognized text and confidence scores
                rec_texts = result_item.get('rec_texts', [])
                rec_scores = result_item.get('rec_scores', [])
                
                # Combine text and confidence
                for i, text in enumerate(rec_texts):
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    
                    # Only keep text with high confidence
                    if confidence > 0.5 and text.strip():
                        texts.append(text)
            #print(texts)
            return ' '.join(texts).strip()
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return ""
    
    def process_mmsd1_dataset(self, data_dir: str, output_dir: str):
        """
        Process MMSD1 dataset
        
        Args:
            data_dir (str): MMSD1 data directory
            output_dir (str): Output directory
        """
        logger.info("Starting to process MMSD1 dataset...")
        
        for split in ['train', 'test', 'valid']:
            file_path = os.path.join(data_dir, f"{split}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            output_file = os.path.join(output_dir, f"mmsd1_{split}_ocr.json")
            ocr_results = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"Processing MMSD1 {split} set, total {len(lines)} items")
            
            for line in tqdm(lines, desc=f"MMSD1 {split}"):
                try:
                    # Parse data line
                    data = ast.literal_eval(line.strip())
                    tweet_id = data[0]
                    text = data[1]
                    label = data[2]
                    
                    # Build image path
                    image_path = os.path.join("data", "dataset_image", f"{tweet_id}.jpg")
                    
                    # Extract OCR text
                    ocr_text = ""
                    if os.path.exists(image_path):
                        ocr_text = self.extract_text_from_image(image_path)
                    
                    # Save result (only keep id and ocr text)
                    result = {
                        "id": tweet_id,
                        "ocr_text": ocr_text
                    }
                    ocr_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing MMSD1 data line: {str(e)}")
                    continue
            
            # Save results to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"MMSD1 {split} set processing completed, results saved to: {output_file}")
    
    def process_mmsd2_dataset(self, data_dir: str, output_dir: str):
        """
        Process MMSD2 dataset (same format as MMSD1)
        
        Args:
            data_dir (str): MMSD2 data directory
            output_dir (str): Output directory
        """
        logger.info("Starting to process MMSD2 dataset...")
        
        for split in ['train', 'test', 'valid']:
            file_path = os.path.join(data_dir, f"{split}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            output_file = os.path.join(output_dir, f"mmsd2_{split}_ocr.json")
            ocr_results = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"Processing MMSD2 {split} set, total {len(lines)} items")
            
            for line in tqdm(lines, desc=f"MMSD2 {split}"):
                try:
                    # Parse data line
                    data = ast.literal_eval(line.strip())
                    tweet_id = data[0]
                    text = data[1]
                    label = data[2]
                    
                    # Build image path
                    image_path = os.path.join("data", "dataset_image", f"{tweet_id}.jpg")
                    
                    # Extract OCR text
                    ocr_text = ""
                    if os.path.exists(image_path):
                        ocr_text = self.extract_text_from_image(image_path)
                    
                    # Save result (only keep id and ocr text)
                    result = {
                        "id": tweet_id,
                        "ocr_text": ocr_text
                    }
                    ocr_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing MMSD2 data line: {str(e)}")
                    continue
            
            # Save results to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"MMSD2 {split} set processing completed, results saved to: {output_file}")
    
    def process_mmsd3_dataset(self, data_dir: str, output_dir: str):
        """
        Process MMSD3 dataset - Write all split data to a single file, writing after each item is processed (standard JSON format)
        
        Args:
            data_dir (str): MMSD3 data directory
            output_dir (str): Output directory
        """
        logger.info("Starting to process MMSD3 dataset...")
        
        # Create unified output file
        output_file = os.path.join(output_dir, "mmsd3_all_ocr.json")
        total_processed = 0
        ocr_results = []
        
        for split in ['train', 'val', 'test']:
            file_name = f"{split}_data.json"
            file_path = os.path.join(data_dir, file_name)
            
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            # Read JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            
            logger.info(f"Processing MMSD3 {split} set, total {len(data_list)} items")
            
            for item in tqdm(data_list, desc=f"MMSD3 {split}"):
                try:
                    item_id = item.get('id')
                    images = item.get('images', [])
                    text = item.get('text', '')
                    label = item.get('label', '')
                    star_rating = item.get('star_rating', None)
                    source = item.get('source', '')
                    
                    # Process OCR for multiple images - one record per image
                    for img_idx, img_path in enumerate(images):
                        # Convert relative path to absolute path
                        if img_path.startswith('./'):
                            full_img_path = os.path.join("data", img_path[2:])
                        else:
                            full_img_path = img_path
                        
                        ocr_text = ""
                        if os.path.exists(full_img_path):
                            ocr_text = self.extract_text_from_image(full_img_path)
                        
                        # Create separate record for each image
                        result = {
                            "id": f"{item_id}_{img_idx + 1}",  # Generate unique ID for each image
                            "ocr_text": ocr_text
                        }
                        
                        # Add to results list
                        ocr_results.append(result)
                        total_processed += 1
                        
                        # Write to file after processing each item (overwrite with complete JSON array)
                        with open(output_file, 'w', encoding='utf-8') as f_out:
                            json.dump(ocr_results, f_out, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error processing MMSD3 data item: {str(e)}")
                    continue
        
        logger.info(f"MMSD3 dataset processing completed, total {total_processed} records processed, results saved to: {output_file}")
    
    def run_ocr_extraction(self, base_data_dir: str = "data", output_dir: str = "ocr_results"):
        """
        Run complete OCR extraction process
        
        Args:
            base_data_dir (str): Base data directory
            output_dir (str): Output directory
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process datasets (skip MMSD2 as it's a subset of MMSD1)
        datasets = [
            #("MMSD1", self.process_mmsd1_dataset),
            ("MMSD3", self.process_mmsd3_dataset)
        ]
        
        for dataset_name, process_func in datasets:
            dataset_dir = os.path.join(base_data_dir, dataset_name)
            
            if os.path.exists(dataset_dir):
                logger.info(f"Starting to process {dataset_name} dataset...")
                try:
                    process_func(dataset_dir, output_dir)
                    logger.info(f"{dataset_name} dataset processing completed")
                except Exception as e:
                    logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"{dataset_name} dataset directory does not exist: {dataset_dir}")
        
        logger.info("OCR extraction completed! Note: MMSD2 was skipped as it's a subset of MMSD1.")

def main():
    """Main function"""
    logger.info("Starting OCR text extraction task...")
    
    try:
        # Initialize OCR extractor
        # If no GPU or insufficient GPU memory, set use_gpu=False
        # Set lang='en' to recognize English
        extractor = OCRExtractor(use_gpu=True, lang='en')
        
        # Run OCR extraction
        extractor.run_ocr_extraction()
        
        logger.info("OCR text extraction task completed!")
        
    except Exception as e:
        logger.error(f"Error occurred during OCR extraction: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
