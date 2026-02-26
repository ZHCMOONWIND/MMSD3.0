# MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection

> 🎉 **CVPR 2026** | Official Implementation

Welcome to the official repository of **MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection**, accepted at **CVPR 2026**.

## ✨ Overview

**MMSD3.0** is the first benchmark that explicitly targets **multi-image multimodal sarcasm detection** in real-world settings.

- 🖼️ **Real-world multi-image benchmark:** built from **X/Twitter** and **Amazon reviews**, with **10,000+ samples**, each containing **2–4 images**
- 🧠 **Richer sarcastic cues:** preserves **longer text**, **emoji signals**, and **OCR-rich visual text** for more realistic multimodal reasoning
- 🔍 **Beyond single-image sarcasm:** focuses on sarcasm triggered by **cross-image relations** and **text–image incongruity**
- 🏗️ **Accompanied by CIRM:** a dedicated **Cross-Image Reasoning Model** with **Dual-Stage Bridging** and **Relevance-Guided Fusion**

## 📁 Repository Structure

```text
data/
  dataset_image/                         # Image root for MMSD1.0 & MMSD2.0
  images/                                # Image root for MMSD3.0 (Amazon & X platform)
  MMSD1/{train.txt, valid.txt, test.txt}
  MMSD2/{train.txt, valid.txt, test.txt}
  MMSD3/{train_data_opensource.json, val_data_opensource.json, test_data_opensource.json}

models/                                  # Model checkpoints

ocr_results/
  mmsd1_train_ocr.json
  mmsd1_valid_ocr.json
  mmsd1_test_ocr.json
  mmsd3_all_ocr.json                     # OCR results for MMSD3.0

results/                                 # Training logs & evaluation results

src/
  config.py                              # Configuration and hyperparameters
  train.py                               # Main training script
  model.py
  text_encoder.py
  vision_encoder.py
  trainer.py
  fast_data_loader.py

download_tweets_from_api.py              # X platform data download script
extract_ocr_text.py                      # OCR extraction script
```

## 📦 Dataset Setup

### MMSD1.0

Download from:
[https://github.com/headacheboy/data-of-multimodal-sarcasm-detection](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)

### MMSD2.0

Download from:
[https://github.com/JoeYing1019/MMSD2.0](https://github.com/JoeYing1019/MMSD2.0)

### MMSD3.0

#### 🛒 Amazon Reviews Data

**Data source:**  
[https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)

**Image download:**

1. Download images from:  
   [https://drive.google.com/file/d/1BdhwkK_vGC13IGnaKk7LZ7V934n95ayN/view?usp=drive_link](https://drive.google.com/file/d/1BdhwkK_vGC13IGnaKk7LZ7V934n95ayN/view?usp=drive_link)
2. Extract `images.zip` to `data/images/`

#### 🐦 X Platform Data

Due to X platform policies, we provide **post links** instead of redistributing raw post content.

To request access to the X-platform portion of MMSD3.0, please first complete the access form below:  
[https://forms.office.com/r/g3TZBSssqt](https://forms.office.com/r/g3TZBSssqt)

After approval, you may reconstruct the dataset by running the download script to retrieve post text and images:

```bash
python download_tweets_from_api.py
```

### 🔍 OCR Extraction

After downloading all images, run the following script to extract OCR text:

```bash
python extract_ocr_text.py
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the datasets

Set up the datasets by following the instructions above.

### 3. Configure training

Edit `src/config.py` to:

* Set `Config.dataset` to `"MMSD1"`, `"MMSD2"`, or `"MMSD3"`
* Adjust data paths and hyperparameters as needed

### 4. Run training

```bash
python src/train.py
```

📌 Notes:

* Model checkpoints will be saved to `models/`
* Training logs and evaluation results will be saved to `results/`

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{mmsd3_cvpr2026,
  title={MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection},
  author={Anonymous},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📬 Contact

For any questions, suggestions, or issues, please contact:
**[zhaohaochen@iie.ac.cn](mailto:zhaohaochen@iie.ac.cn)**
