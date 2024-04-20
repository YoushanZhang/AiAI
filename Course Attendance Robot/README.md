# Course Attendance Robot
<div align="center">
    <a><img width="720" src="images/face-recognition-attendance-system.jpg" alt="soft"></a>
</div>

## Table of Contents

- [Course Attendance System with Facial Recognition](#course-attendance-system-with-facial-recognition)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
      - [Project Description](#project-description)
      - [Website](#website-screenshots)
  - [Datasets](#datasets)
      - [Raw Images](#raw-images-data-collection)
      - [Processed Images ](#processed-images)
  - [Method](#method)
      - [RetinaFace](#retinaface)
      - [Face Recognition](#face-recognition)
  - [Results](#results)
  - [Technical Information](#technical-information)
  - [Benefits](#benefits)
  - [Applications](#applications)
  - [Citations](#citations)
   
## Introduction

### Project Description

This project introduces a comprehensive system for managing attendance, leveraging facial detection and recognition technologies to identify individual students and register their attendance. The system is developed using Python, RetinaFace, Face_Recognition, and OpenCV. It offers an efficient and automated solution for monitoring attendance across diverse settings such as educational institutions and workplaces.

Users can also use a web-based graphical interface, to upload classroom images to perform automatic attendance and save the results directly into a CSV file.

### Website Screenshots

<figure align="center"> 
  <img src="images/AddCourse.png" alt="drawing" height="400"/>
  <figcaption>Screen to Facilitate Course Addition</figcaption>
</figure>

<figure align="center"> 
  <img src="images/CaptureAttendance.jpg" alt="drawing" height="400"/>
  <figcaption>Screen to Capture Course Attendance</figcaption>
</figure>

<figure align="center"> 
  <img src="images/RecordAttendance.jpg" alt="drawing" height="400"/>
  <figcaption>Screen to Modify, Save or Download Attendance CSV.</figcaption>
</figure>


## Datasets:

### Raw Images Data Collection:
To assess the efficacy of the proposed system, a dataset containing thirty-six students was compiled. Facial images of volunteer students are captured using a laptop web camera and saved in a folder. The dataset utilized in this investigation encompasses 100 images for a sample size of thirty-six (36) students totaling 3,600 images, exhibiting a variety of poses including front, left, and right facial features. The images were standardized to dimensions of 112 pixels in height and 112 pixels in width. Additionally, 69 class group photographs were captured to evaluate the individual recognition of students' faces within group settings. 

### Raw Images:
<figure align="center"> 
  <img src="images/training112.png" alt="drawing" height="400"/>
  <figcaption>Raw Images of Dimension 112x112</figcaption>
</figure>

### Processed Images:
<figure align="center"> 
  <img src="images/training50.png" alt="drawing" height="400"/>
  <figcaption>Processed Images Ready for Embeddings 50x50</figcaption>
</figure>

## Method:

### RetinaFace:

RetinaFace is a robust single-stage face detector that performs pixel-wise face localization on various scales of faces by taking advantage of joint extra-supervised and self-supervised multi-task learning.

### Retinaface Output Screenshots:
<figure align="center"> 
  <img src="images/15_Detection.jpg" alt="drawing" height="720"/>
  <figcaption>RetinaFace detection output on Classroom Image</figcaption>
</figure>


### Face Recognition:
Face Recognition is a Python library that can be used to recognize and manipulate faces. It is built on top of Dlib modern C++ toolkit that contains machine-learning algorithms and tools for creating complex software in C++ to solve real-world problems. As a part of facial recognition, after the facial images have been extracted, cropped, resized, and often converted to grayscale, the face recognition algorithm takes on the task of identifying features that most accurately represent the image.

### Recognized Faces:
<figure align="center"> 
  <img src="images/15_predicted.jpg" alt="drawing" height="720"/>
  <figcaption>Face Recognition on group image</figcaption>
</figure>

## Results

| Detection           | Recognition           | Image Size | Cropped Embeddings | Detection Accuracy (%) | Recognition Accuracy (%) | Group Image | Recognition Using |
|---------------------|-----------------------|------------|--------------------|------------------------|-------------------------|-------------|-------------------|
| Face Mesh           | Deep Neural Network  | -          | -                  | -                      | 94.2                    | No          | Image             |
| **RetinaFace**          | **Face Recognition **    | **50x50**      | **Yes**                | **99.4**                   | **90.3**                    | **Yes**         | **Image**            |
| RetinaFace          | Face Recognition     | 50x50      | No                 | 99.4                   | 88.26                   | No          | Image             |
| RetinaFace          | Face Recognition     | 112x112    | No                 | 99.16                  | 85.3                    | No          | Image             |
| RetinaFace          | Face Recognition     | 600x600    | No                 | 98.6                   | 85.1                    | No          | Image             |
| Face Recognition    | Face Recognition    | 50x50      | No                 | 65.53                  | 72.3                    | No          | Image             |
| Face Recognition    | Face Recognition    | 112x112    | No                 | 65.53                  | 70.53                   | No          | Image             |
| Yolo9               | Face Recognition     | 112x112    | No                 | 51.68                  | 58.41                   | No          | Image             |
| RetinaFace          | haarcascade_frontalface | 50x50  | No                 | 99.4                   | 45.2                    | No          | Image             |
| Dlib                | Face Recognition     | 112x112    | No                 | 91                     | 86                      | Yes         | Image             |
| InsightFace         | InsightFace          | 1200x1600  | No                 | 99                     | 88                      | Yes         | Image             |
| MTCNN               | Facenet              | -          | -                  | -                      | 100                     | No          | Video             |
| HOG                 | Dlib                 | -          | -                  | -                      | 94                      | No          | Video             |
| Viola-Jones         | LBPH                 | -          | -                  | -                      | 91                      | No          | Video             |
| Haar cascade + DoG filtering | LBPH         | -          | No                 | 98.36                  | 87                      | No          | Video             |



## Technical Information:

- **Programming Language**: Python
- **Face Detection**: RetinaFace
- **Face Recognition**: Face Recognition.
- **Computer Vision Library**: OpenCV

## Benefits:

- **Effortless Efficiency**: Automates attendance, saving time and resources.
- **Increased Accuracy**: Reduces human error compared to manual roll calls.
- **Enhanced Convenience**: Provides a faster and more user-friendly approach for everyone.
- **Flexible Scalability**: Adapts to accommodate different group sizes.

## Applications:

- **Educational Institutions**: Streamlines attendance management and ensures accurate student records.
- **Workplaces**: Simplifies employee attendance tracking and supports flexible work arrangements.
- **Access Control Systems**: Offers an additional layer of security by integrating face recognition for entry.

## Citations:
1. https://arxiv.org/abs/1905.00641
2. https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.timedynamo.com%2Fblog%2Fface-recognition-attendance-system&psig=AOvVaw2FkF7iZtn_xnR0WqiLOgx8&ust=1713675455940000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMjdmdmA0IUDFQAAAAAdAAAAABAS




















