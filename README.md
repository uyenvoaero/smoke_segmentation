# Smoke Segmentation

A deep learning project for smoke segmentation in images and videos using the U-Net model with a ResNet50 backbone, built on the [segmentation_models](https://github.com/qubvel/segmentation_models) library. The project supports training, inference, and visualization of smoke detection results.

## Features
- **Model**: U-Net with ResNet50 backbone for binary smoke segmentation.
- **Data Processing**: Handles image and video inputs with preprocessing and augmentation using Albumentations.
- **Training**: Customizable training pipeline with Dice + Focal loss and IoU/F-score metrics.
- **Inference**: Processes images and videos, overlaying predicted masks with visualization.
- **Output**: Saves results as images or videos with ground truth and predicted masks.

## Sample Inference Results
Below are example outputs showcasing the model's smoke segmentation performance:

1. **Smoke detection on a test image with ground truth and predicted mask**  
   ![Sample Result 1](https://raw.githubusercontent.com/uyenvoaero/smoke_segmentation/main/output/smoke_detection_test_5.png)

2. **Smoke segmentation on a test video**  
   ![Sample Result 2](https://raw.githubusercontent.com/uyenvoaero/smoke_segmentation/main/output/smoke_2_output.gif)

## Usage
1. Clone the repository: `git clone https://github.com/uyenvoaero/smoke_segmentation.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare dataset with images and masks in the specified directory structure.
4. Run training: `python train.py`
5. Perform inference: `python inference.py`

## Requirements
- Python 3.8+
- TensorFlow/Keras
- segmentation_models
- Albumentations
- OpenCV
- NumPy
- Matplotlib

## License
MIT License
