# Handwritten Digit & Alphabet Recognition System

A deep learning–based web application for recognizing handwritten digits and alphabet characters using a Convolutional Neural Network (CNN) trained on the EMNIST dataset. The application integrates image preprocessing techniques using OpenCV and provides real-time predictions through a Streamlit interface.

---

## Overview

This project demonstrates the implementation of an end-to-end handwritten character recognition pipeline, covering:

- Data preprocessing
- CNN model development and training
- Image enhancement and normalization
- Real-time inference
- Web application deployment

The system is capable of recognizing handwritten digits and alphabet characters from uploaded images with an overall model accuracy of approximately **90%** on the EMNIST dataset.

---

## Key Features

- Handwritten digit and alphabet recognition
- Convolutional Neural Network (CNN) implementation
- Image preprocessing using OpenCV
- Real-time predictions through Streamlit
- Support for uploaded handwritten images
- Model persistence for inference **without retraining**
- Clean and interactive user interface

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Web Application Framework | Streamlit |

---

## Dataset

The model is trained on the **EMNIST (Extended MNIST)** dataset, an extension of the MNIST dataset containing handwritten digits and alphabet characters.

### Dataset Characteristics

- Digits: `0–9`
- Alphabet characters: uppercase and lowercase
- Large-scale handwritten character dataset
- Suitable for multi-class image classification tasks

---

## Model Architecture

The recognition model is implemented using a Convolutional Neural Network (CNN) designed for image classification tasks.

### Architecture Components

- Convolutional Layers
- ReLU Activation Functions
- Max Pooling Layers
- Fully Connected Dense Layers
- Softmax Output Layer

The CNN extracts hierarchical spatial features from handwritten character images and performs multi-class classification.

---

## Image Processing Pipeline

Prior to inference, uploaded images undergo preprocessing using OpenCV to improve prediction accuracy.

### Processing Steps

1. Convert image to grayscale
2. Apply binary thresholding
3. Detect handwritten region
4. Remove unnecessary background noise
5. Resize image to model input dimensions
6. Center the handwritten character
7. Normalize pixel intensity values
8. Pass processed image to CNN model

---

## Model Performance

| Metric | Value |
|---|---|
| Dataset | EMNIST |
| Model Type | CNN |
| Accuracy | ~90% |

---

## Application Deployment

The application is deployed using Streamlit, enabling users to interact with the trained model through a web interface.

---

## Installation & Setup

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
├── app.py                    # Streamlit application
├── model/
│   └── handwritten_nb.keras  # Trained CNN model
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Project dependencies
└── README.md
```

---

## Future Enhancements

Potential improvements and extensions include:

- Interactive drawing canvas support
- Improved model accuracy through data augmentation
- Top-k prediction probabilities
- Multi-character recognition
- OCR-based sentence recognition
- Docker containerization
- Cloud deployment support
- Transfer learning implementation

---

## Learning Outcomes

This project demonstrates practical understanding of:

- Deep Learning fundamentals
- Convolutional Neural Networks
- Computer Vision techniques
- Image preprocessing pipelines
- Model deployment workflows
- Real-time machine learning inference

---

## Screenshots


<img width="567" height="164" alt="Screenshot 2026-05-12 at 10 39 25 PM" src="https://github.com/user-attachments/assets/4ba6fe80-cdf3-45a2-83e8-dde2626788af" />
<p align="center">
  <img src="https://github.com/user-attachments/assets/a33a5c6f-d079-4fec-86ff-19201fa2fba5" width="48%" />
  <img src="https://github.com/user-attachments/assets/a6b5c735-cfe1-44aa-ae02-88e23d3090ef" width="25%" />
</p>

---

## Author

**Rageshwer Singh**

Developed using Deep Learning, OpenCV, and Streamlit.
