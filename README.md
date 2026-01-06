# Eye-disease-classifier
# Convolutional Neural Network for Eye Disease Classification

## Project Overview
This project implements a deep learning pipeline designed to classify retinal diseases (e.g., Diabetic Retinopathy vs. Healthy Fundus) using transfer learning. 

The system utilizes a **ResNet-18** architecture pre-trained on ImageNet, fine-tuned with a custom fully connected layer for binary classification. It includes a PyQt6 graphical interface for clinical demonstration purposes.

## Key Features
* **Model Architecture:** ResNet-18 (Residual Neural Network) with modified output layers for medical binary classification.
* **Optimization:** Implements CrossEntropyLoss and Adam Optimization ($lr=0.001$).
* **Interface:** User-friendly desktop GUI built with PyQt6 for real-time image inference.
* **Testing:** Comprehensive unit testing suite for data loading and tensor shape verification.

## Technical Stack
* **Core:** Python 3.9+, PyTorch
* **Data Processing:** NumPy, Pandas, Pillow
* **Visualization/GUI:** PyQt6, Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/AZakieh/Eye-disease-classifier.git](https://github.com/AZakieh/Eye-disease-classifier.git)

2. Install dependencies:
    '''bash
    pip install -r requirements.txt

## Usage
In order to run the training loop, ensure you have a ./data folder with healthy and disease subfolders and run:
    '''bash
    python eye_classifier.py
To launch the inference GUI:
    '''bash
    python gui.py

## Future Improvements
.Integration with the OCTMNIST dataset for robust validation.
.Implementation of Grad-CAM to visualize class activation maps (explainability).