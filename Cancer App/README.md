# Pink Guardian - Early Breast Cancer Detection App



Welcome to the **Pink Guardian** GitHub repository, where we integrate cutting-edge machine learning technology into mobile health applications to improve early breast cancer (bc) detection. This project showcases the development of the BC-InceptionV3 model, its training on Kaggle, conversion to TensorFlow Lite, and deployment in the Pink Guardian app.

## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Model Development](#model-development)
  - [Dataset Overview](#dataset-overview)
  - [Training on Kaggle](#training-on-kaggle)
  - [Conversion to TensorFlow Lite](#conversion-to-tensorflow-lite)
- [App Details](#app-details)
  - [Integration with Mobile App](#integration-with-mobile-app)
  - [Download the App](#download-the-app)
- [Results](#results)
- [Technical Information](#technical-information)
-  [Benefits](#benefits)
- [Applications](#applications)
- [Citations](#citations)

## Introduction

Breast cancer remains a major global health challenge. Early detection significantly increases the chances of successful treatment. Pink Guardian utilizes advanced AI to provide accessible diagnostic tools directly through a mobile application, making early screenings more available and reducing barriers to access.

## Project Description

This project involves the creation of a comprehensive system that employs the BC-InceptionV3 model trained to classify mammogram images as benign or malignant. The model is trained in Python notebooks on Kaggle, converted to TensorFlow Lite, and then integrated into an Android app for easy user accessibility.

## Model Development

### Dataset Overview


<img src="(https://private-user-images.githubusercontent.com/26533517/326626822-5d697a54-a483-4628-9a5e-29627029e4f9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQ0MzcyMDcsIm5iZiI6MTcxNDQzNjkwNywicGF0aCI6Ii8yNjUzMzUxNy8zMjY2MjY4MjItNWQ2OTdhNTQtYTQ4My00NjI4LTlhNWUtMjk2MjcwMjllNGY5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDMwVDAwMjgyN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ0OWEyYzgyMTIzZjZmZjNiOGUwNTVmMTkzNWU2MzQyMTg2OTRhNTFkNjUyNjNjZWZhYmI1ZjY5YTEzMjEzM2MmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.hZ81Gh7twal57EtN17jELyF3YwcQjDTZamaxGzu8Mzg)" width="807" alt="dataset">

The BC-InceptionV3 model is trained on high-quality mammographic images that provide a diverse basis for learning distinguishing features between benign and malignant tumors.


### Training on Kaggle

The model training is executed in a Python notebook hosted on Kaggle, leveraging its GPU resources to efficiently handle the computational demands of training a deep learning model.

### Conversion to TensorFlow Lite

Post-training, the model is converted to TensorFlow Lite to optimize performance for mobile devices, ensuring that the app runs smoothly across a wide range of smartphones.

## App Details

### Integration with Mobile App

The TensorFlow Lite model is integrated into the Pink Guardian mobile app, providing users with real-time analysis of mammogram images. This integration allows for immediate preliminary diagnostics to be made accessible on user's devices.

### Download the App

Scan the QR code below to download the **Pink Guardian** app directly to your device:


<img src="https://private-user-images.githubusercontent.com/26533517/326626110-fb8ed096-e201-4808-b57f-a4fac201e2c9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQ0MzY4NjksIm5iZiI6MTcxNDQzNjU2OSwicGF0aCI6Ii8yNjUzMzUxNy8zMjY2MjYxMTAtZmI4ZWQwOTYtZTIwMS00ODA4LWI1N2YtYTRmYWMyMDFlMmM5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDMwVDAwMjI0OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTkwMTgwMzM0NzEwZDdmMTFlMTIwYWNkODkxZWQ0YzIyMGQ0YmMzMjEwZWM0OTJhY2JmN2ExMTkxZjRiYjg3NzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.NlUV-e3o6lgR7renFWr5uiy6wWiAQPN_CnqY0cMxgBI" alt="QR Code for Pink Guardian App" width="200"/>


## Results

The application of the BC-InceptionV3 model within the Pink Guardian app demonstrates high accuracy in detecting breast cancer signs from mammograms, providing a reliable tool for early detection.
Our comprehensive evaluation of various models reveals that the BC Inception V3 significantly outperforms its counterparts with a remarkable accuracy of 99.49% during training and 78.70% in validation. Here are the detailed results:

| Model              | Loss    | Accuracy (%) | Validation Loss | Validation Accuracy (%) |
|--------------------|---------|--------------|-----------------|------------------------|
| ResNet50           | 0.6895  | 55.18        | 0.6986          | 51.79                  |
| ResNet101          | 0.6839  | 56.53        | 0.7019          | 51.79                  |
| EfficientNet B0    | 0.6887  | 54.76        | 0.6943          | 51.79                  |
| Xception           | 0.4547  | 77.17        | 0.7171          | 65.48                  |
| Inception          | 0.5508  | 70.85        | 0.5847          | 67.26                  |
| Fine-tuned Xception| 0.5813  | 69.50        | 0.6602          | 62.50                  |
| EfficientNet B7    | 0.6887  | 54.76        | 0.6942          | 51.79                  |
| Nesnet Mobile      | 0.5134  | 73.46        | 0.6881          | 61.31                  |
| Custom Xception    | 0.5705  | 71.78        | 0.6396          | 64.88                  |
| BC Inception V3    | 0.1454  | 99.49        | 0.6645          | 78.70                  |

## Technical Information

- **Programming Language**: Python
- **Technologies Used**: TensorFlow Lite, Kaggle, Android Studio
- **Model**: Custom BC-InceptionV3

## Benefits

The Pink Guardian app powered by the Custom Inception V3 model offers several key benefits:

- **Early Detection**: Facilitates the early detection of breast cancer, which is crucial for successful treatment outcomes.
- **High Accuracy**: Delivers high accuracy in mammogram analysis, reducing the likelihood of false negatives and positives.
- **Accessibility**: Provides a readily accessible breast cancer screening tool, directly on the user's smartphone.
- **Cost-Effective**: Offers a cost-effective alternative to traditional diagnostic methods, potentially reducing healthcare expenses.
- **Time-Efficient**: Saves time for both patients and healthcare professionals through swift analysis and diagnosis.
- **User-Friendly**: Features an easy-to-use interface that simplifies the process of uploading and analyzing mammogram images.
- **Privacy-First**: Ensures user privacy by processing sensitive medical images directly on the device without the need to upload them to external servers.
- **Educational**: Acts as an educational resource by providing users with information on breast cancer and the importance of early detection.

## Applications

The applications of the Pink Guardian app span across various sectors, including but not limited to:

- **Healthcare Sector**: Clinics and hospitals can use the app as a preliminary screening tool, providing quick guidance to patients before further medical consultation.
- **Remote Areas**: In remote regions where medical resources are scarce, Pink Guardian serves as an essential tool for initial screening, potentially bridging gaps in healthcare accessibility.
- **Research and Education**: Educational institutions and research facilities can leverage the app for studying breast cancer patterns and raising awareness.
- **Telemedicine**: The app can be integrated into telemedicine services, enabling healthcare professionals to conduct preliminary assessments remotely.
- **Personal Health Monitoring**: Individuals can use the app for personal health monitoring, staying informed about their breast health and seeking medical advice when necessary.
- **Corporate Health Programs**: Employers can incorporate Pink Guardian into their health and wellness programs to promote the wellbeing of their workforce.


## Citations

Relevant studies and references that have informed the development of the BC-InceptionV3 model and its implementation in Pink Guardian can be found here:

- [A curated mammography data set for use in computer-aided detection and diagnosis research](https://www.nature.com/articles/sdata2017177)
- [Kaggle for Data Science and Machine Learning](https://www.kaggle.com/)
- [TensorFlow Lite for Mobile Deployment](https://www.tensorflow.org/lite)

For more detailed references, please refer to the included citation file or section within the project documentation.
