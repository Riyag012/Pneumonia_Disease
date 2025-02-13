# Pneumonia Detection using Deep Learning

This project utilizes a **VGG16-based deep learning model** to detect **pneumonia** from chest X-ray images. The model is deployed using **Streamlit** and is available online.

🔗 **Live Demo**: [Pneumonia Detection Web App](https://pneumonia-disease-detection.streamlit.app/)

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation & Setup](#installation--setup)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

---

## Overview
Pneumonia is a severe lung infection that requires early detection for effective treatment. This project aims to automate pneumonia detection from X-ray images using **deep learning** and **transfer learning (VGG16)**.

## Dataset
The dataset is sourced from **Kaggle's Chest X-ray dataset**, which contains labeled X-ray images for **Normal** and **Pneumonia** cases.

- **Train Set**: Used for training the model
- **Validation Set**: Used to fine-tune hyperparameters
- **Test Set**: Used to evaluate the model performance

📌 **Dataset Link**: [Chest X-ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Model Architecture
The model is built using **VGG16 (pretrained on ImageNet)**, with the final layers modified for binary classification.

- **Feature Extractor**: VGG16 (pretrained, frozen layers)
- **Classifier**: GlobalAveragePooling + Dense layers
- **Activation**: ReLU & Sigmoid
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam

## Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

### 2️⃣ Create & Activate Virtual Environment
```sh
python -m venv env
# Activate on Windows
env\Scripts\activate
# Activate on Mac/Linux
source env/bin/activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 🚀 Model Training
To train the model, run:
```sh
python train.py
```
This will train the model on the dataset and save the trained model.

## Deployment
The model is deployed using **Streamlit** on **Streamlit Cloud**.

To deploy locally, run:
```sh
streamlit run app.py
```

To deploy online, follow these steps:
1. Push your repository to GitHub.
2. Create an account on [Streamlit Cloud](https://share.streamlit.io/).
3. Deploy by selecting your GitHub repository.

🔗 **Live Deployment**: [Pneumonia Detection Web App](https://pneumonia-disease-detection.streamlit.app/)

## 🔧 Usage
1. Open the [web app](https://pneumonia-disease-detection.streamlit.app/)
2. Upload a chest X-ray image.
3. The model predicts whether the image is **Normal** or indicates **Pneumonia**.
4. The result is displayed with a confidence score.

## Results
The model achieved:
- **Training Accuracy**: 91.5%
- **Validation Accuracy**: 84.78%
- **Validation Loss**: 0.3523

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **PIL**
- **Matplotlib**
- **Google Colab (for training)**

## Future Enhancements
- Improve accuracy using more advanced architectures like **EfficientNet**.
- Deploy on **Hugging Face Spaces** or **Flask-based Web App**.
- Implement **Grad-CAM visualization** to explain model predictions.

