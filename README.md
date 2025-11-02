# PAN Card Tampering Detection
## Problem Statement

The goal of this project is to build a robust system that can automatically detect whether a given PAN (Permanent Account Number) card image has been tampered with or is authentic. This is crucial for preventing identity fraud and ensuring document authenticity in financial systems.

### Dataset Information

- **Total Images:** 999
- **Real Images:** 647 (64.8%)
- **Fake Images:** 352 (35.2%)
- **Image Format:** Various resolutions, primarily JPEG/PNG
- **Challenge:** Class imbalance with approximately 1.8:1 ratio

---

## Approach

### Model Architecture

I experimented with multiple deep learning architectures:
1. **EfficientNet-B4** - Initial experiments showed poor performance (61.83% accuracy), likely due to learning rate and augmentation issues
2. **ConvNeXt-Base** - Strong performer with modern CNN architecture
3. **Swin Transformer** - Vision transformer approach with hierarchical feature maps

### Training Strategy
**Cross-Validation Setup:**
- 5-fold stratified cross-validation to ensure robust evaluation
- Maintained class distribution across all folds
- Trained separate models for each fold

**Hyperparameters:**
- Image Size: 384x384
- Batch Size: 16
- Optimizer: AdamW
- Learning Rate: 1e-4
- Loss Function: Focal Loss (to handle class imbalance)
- Epochs: 30 with early stopping

**Data Augmentation:**
- Horizontal flips
- Random rotation (up to 10 degrees)
- Brightness and contrast adjustments
- Gaussian noise and blur
- JPEG compression simulation

---

## Results

### Individual Model Performance

#### ConvNeXt-Base
After 5-fold cross-validation:
- **Average Accuracy:** 92.49%
- **Average Precision:** 0.8970
- **Average Recall:** 0.8958
- **Average F1 Score:** 0.8957

This model showed consistent performance across all folds with minimal variance, making it a reliable choice.

#### Swin Transformer
Performance across 5 folds:
- **Average Accuracy:** 92.49%
- **Average Precision:** 0.9009
- **Average Recall:** 0.8999
- **Average F1 Score:** 0.8997

The transformer-based approach matched ConvNeXt's performance with slightly better precision, indicating fewer false positives.

#### EfficientNet-B4
Initial results were disappointing:
- **Average Accuracy:** 61.83%
- **Average Precision:** 0.5837
- **Average Recall:** 0.5592
- **Average F1 Score:** 0.5639

### Best Performing Models

After analyzing individual fold results, I identified four models that consistently outperformed others:
1. convnext_base_fold2
2. convnext_base_fold4
3. swin_base_patch4_window12_384_fold2
4. swin_base_patch4_window12_384_fold3

These four models were selected for the final ensemble based on their validation performance and consistency.

### Ensemble Performance

Using simple averaging of the four best models:
- **Ensemble Accuracy:** ~93-94%
- **Precision:** ~0.92-0.93
- **Recall:** ~0.91-0.92
- **F1 Score:** ~0.92

The ensemble approach provided more stable predictions compared to individual models and reduced the impact of any single model's errors.

---

## Key Features Implemented

### 1. Multi-Model Ensemble

Combined predictions from multiple architectures to improve robustness. Each model brings different inductive biases, which helps in catching various types of tampering.

### 2. GradCAM Visualization

Implemented Gradient-weighted Class Activation Mapping to understand what regions of the PAN card the models focus on when making predictions. This provides interpretability and helps identify whether the model is learning relevant features.

### 3. OCR Integration

Added optical character recognition to extract and validate text from PAN cards:

- Primary: EasyOCR (better for document images) and Tesseract OCR
- Features extracted: Text content, bounding boxes, confidence scores
- PAN number format validation using regex patterns


---

## Technical Implementation

### Core Technologies

- **Deep Learning:** PyTorch, timm (for pretrained models)
- **Computer Vision:** OpenCV, Albumentations
- **OCR:** EasyOCR, Tesseract
- **Machine Learning:** XGBoost, LightGBM (for meta-learning)

### Training Environment

- Platform: Kaggle Notebooks
- GPU: 2x NVIDIA T4
- Training Time: Approximately 2-3 hours per model
- Total Training Time: ~15-20 hours for all experiments

---

## Notebook

Notebook Link - https://www.kaggle.com/code/aditisalvi04/pan-card-tampering

Notebook Output Link - https://www.kaggle.com/code/aditisalvi04/pan-card-tampering?scriptVersionId=272716121

---

## Output 

Fake PAN card image output
<img width="5107" height="4476" alt="analysis_Tampered_141 jpg" src="https://github.com/user-attachments/assets/83f09785-a1a9-47fe-8bda-10b19eed3546" />


Real PAN card image output
<img width="4219" height="4476" alt="analysis_01271234470012_jpg rf b91eb1b2191522fe9242eccf0be9bca5 jpg" src="https://github.com/user-attachments/assets/a4a4179f-bdad-4ec6-87ff-e3512704df2f" />



---

							                     Built with 💜 by Aditi Salvi
