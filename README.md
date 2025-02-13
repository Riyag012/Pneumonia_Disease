# Pneumonia Detection using Deep Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Details](#training-details)
5. [Installation & Usage](#installation--usage)
6. [Results](#results)
7. [File Structure](#file-structure)
8. [Future Improvements](#future-improvements)
9. [Acknowledgments](#acknowledgments)

## Project Overview
This project aims to develop a deep learning model for detecting pneumonia from chest X-ray images using a convolutional neural network (CNN).
The VGG16 model is used as a feature extractor with a custom classification head. The goal is to assist healthcare professionals in diagnosing pneumonia
effectively and accurately.

## Dataset
The dataset used consists of chest X-ray images categorized into two classes:
- **Normal:** Healthy lung images
- **Opacity/Pneumonia:** Lungs infected with pneumonia

The dataset is sourced from [Kaggle's Chest X-Ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and preprocessed before training.

## Model Architecture
The model is built using **Transfer Learning** with the **VGG16** architecture:
- Pretrained **VGG16** model (without top layers)
- Custom **fully connected layers** added for classification
- **Binary classification** (Pneumonia vs. Normal)
- **Adam optimizer** used for better convergence

## Training Details
- **Batch Size:** 16
- **Epochs:** 20
- **Optimizer:** Adam
- **Learning Rate:** 1e-4
- **Class Imbalance Handling:** Weighted loss applied
- **Training Accuracy:** *91.97*
- **Validation Accuracy:** *87.02*

## Installation & Usage
### Prerequisites
Ensure you have Python and the following dependencies installed:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

### Running the Model
1. Clone this repository:
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook
```
3. Run the `Pneumonia Detection.ipynb` file step by step.

4. To use the trained model, load it using:
```python
from keras.models import load_model
model = load_model('model_vgg16.h5')
```

## Results
The model achieved:
- **Training Accuracy:** *91.97*
- **Validation Accuracy:** *87.02*
- **Validation Loss:** *0.3149*

Visualization of training performance (loss and accuracy curves) is available in the notebook.

## File Structure
```
├── Pneumonia Disease
    ├── Pneumonia Detection.ipynb  # Jupyter Notebook with code
    ├── model_vgg16.h5             # Trained model file
    ├── dataset/                   # Dataset (Not included, download separately)
    └── README.md                  # Project documentation
```

## Future Improvements
- Use **more complex architectures** like EfficientNet for better accuracy.
- **Data Augmentation** to improve generalization.
- Hyperparameter tuning for further optimization.
- Deployment as a web app for real-time predictions.

## Acknowledgments
- **Kaggle Chest X-ray Dataset** for providing the dataset.
- **TensorFlow/Keras** for deep learning tools.
- **Open-source contributors** for making resources available.


