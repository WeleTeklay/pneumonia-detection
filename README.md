## 🩺 Pneumonia Detection from Chest X-rays (EfficientNetB0)
This project implements a high-performance deep learning pipeline for the automated binary classification of Pneumonia from Chest X-ray (CXR) images. It utilizes EfficientNetB0 via Transfer Learning and includes an interactive Streamlit web application with live Grad-CAM model explainability.

## Technical Highlights
Transfer Learning (EfficientNetB0): The model uses EfficientNetB0 pre-trained on ImageNet as a frozen feature extractor, accelerating convergence.

Class Weighting: Dynamically computed class weights are applied during training to effectively handle the imbalance between the NORMAL and PNEUMONIA classes.

Grad-CAM Explainability: The Streamlit app generates a Grad-CAM heatmap overlay, allowing users to visually verify which regions of the lung the model used to make its prediction.

The model achieves a high Recall ($0.95$) for the PNEUMONIA class, which is critical in a medical context to minimize dangerous False Negatives (missed diagnoses).
## Dataset & Structure
The project expects the Chest X-ray dataset to be organized in a standard structure, with the base directory often mounted from Google Drive:

├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
## Technologies Used
Python

TensorFlow / Keras (EfficientNetB0)

NumPy

Matplotlib, Seaborn

Scikit-learn

Streamlit

##  Streamlit
The app.py script provides a user-friendly interface to test the model in real-time.

Prerequisites

A Python environment.

The trained model file (pneumonia_detection_efficientnet.keras) must be in the same directory as app.py.

Installation

Install dependencies
pip install tensorflow numpy pillow matplotlib streamlit scikit-learn

Running the Application

Open your terminal in the project directory and run:

streamlit run app.py

This will open the web application in your browser, allowing you to upload an X-ray image and view the diagnosis and the Grad-CAM explanation.

## Future Improvements
Fine-Tuning: Unfreeze the top layers of EfficientNetB0 and train the entire network with a very low learning rate for potentially higher accuracy.

Advanced Grad-CAM: Implement a smoother or higher-resolution version of the Grad-CAM visualization.

API Deployment: Wrap the model in a REST API (e.g., using FastAPI) for integration into other systems like Electronic Health Records.

Multi-Class Detection: Extend the model to classify different types of pneumonia (e.g., Viral, Bacterial).
