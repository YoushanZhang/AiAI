# Course Attendance Robot
<div align="center">
    <a><img width="720" src="images/face-recognition-attendance-system.jpg" alt="soft"></a>
</div>

## Table of Contents

- [Course Attendance Robot](#course-attendance-robot)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Project Description](#project-description)
    - [Website Screenshots](#website-screenshots)
  - [Datasets:](#datasets)
    - [Raw Images Data Collection](#raw-images-data-collection)
    - [Raw Images](#raw-images)
    - [Processed Images](#processed-images)
  - [Method:](#method)
    - [RetinaFace](#retinaface)
    - [Detected Faces](#detected-faces)
    - [Face Recognition](#face-recognition)
    - [Recognized Faces](#recognized-faces)
  - [Implementation](#implementation)
  - [Results](#results)
  - [Overall Facial Detection and Recognition Results](#overall-facial-detection-and-recognition-results)
  - [Individual Facial Detection and Recognition Results](#individual-facial-detection-and-recognition-results)
  - [Demo](#demo)
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

### Detected Faces:
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

## Implementation:

**1. Train Image Data Collection:**
- Go to the folder 'Student Image Data Collection' and download the 'Dataset_Collecting Images.py' and 'haarcascade_frontalface_default.xml'.
- Create a folder 'Face_Recognition_Images' on your desktop and copy/paste the Python script and haarcascade.xml into the folder. 
- Run the code.
   ```
   Dataset_Collecting Images.py
   ```
python Dataset_Collecting Images.py 
  
**2. Test Image Data Collection:**
- You can collect the Group test images from your mobile preferably Apple iPhone or from good-resolution smartphones.

**3. Generate Embeddings:**
- Download the Generate Embeddings.py file to generate the embedding of the known faces.
- It will create an embedding file that can be used as ground truth embedding for recognition.
   ```
   Generate Embeddings.py
   ```

**4. Face Recognition:**
- Go to our website homepage and select/add the course as per your University courses and Upload the class_list.csv (Sample can be found in the Demo folder).
- Select the date for which attendance should be taken and press the button 'Take Attendance'.
- Now choose the Group Image for which the attendance is to be taken and upload the file and press the button 'Take Attendance'. 
- After which you see the recognized faces on the image along with the attendance displayed which can be manually edited and downloaded into a CSV file.

## Results:

## Overall Facial Detection and Recognition Results

| Detection           | Recognition          | Image Size | Cropped Embeddings | Detection Accuracy (%) | Recognition Accuracy (%) | Group Image Recognition | Recognition Format - Images |
|---------------------|----------------------|------------|--------------------|------------------------|--------------------------|-------------------------|-----------------------------|
| RetinaFace          | Face Recognition    | 50x50      | &#10060;           | 99.4                   | 88.3                     | &#10004;                | &#10004;                    |
| RetinaFace          | Face Recognition    | 112x112    | &#10060;           | 99.2                   | 85.3                     | &#10004;                | &#10004;                    |
| RetinaFace          | Face Recognition    | 600x600    | &#10060;           | 98.6                   | 85.1                     | &#10004;                | &#10004;                    |
| Face Recognition   | Face Recognition    | 50x50      | &#10004;           | 65.5                   | 72.3                     | &#10004;                | &#10004;                    |
| Face Recognition   | Face Recognition    | 112x112    | &#10060;           | 65.5                   | 70.5                     | &#10004;                | &#10004;                    |
| Yolo9               | Face Recognition    | 112x112    | &#10060;           | 51.7                   | 58.4                     | &#10004;                | &#10004;                    |
| RetinaFace          | haarcascade_frontalface | 50x50 | &#10060;           | 99.4                   | 45.2                     | &#10004;                | &#10004;                    |
| InsightFace         | InsightFace         | 1200x1600  | &#10060;           | 99.0                   | 89.2                     | &#10004;                | &#10004;                    |
| haarcascade_frontalface | Face Recognition | 112x112 | &#10060;           | 91.0                   | 86.0                     | &#10004;                | &#10004;                    |
| InsightFace         | InsightFace         | 1200x1600  | &#10060;           | 99.0                   | 88.0                     | &#10004;                | &#10004;                    |
| Haar cascade + DoG filtering | LBPH       | -          | &#10060;           | 98.3                   | 87.0                     | &#10060;                | &#10060;                    |
| RetinaFace[**Ours**]| Face Recognition    | **50x50**  | **&#10004;**       | **99.4**               | **90.3**                 | **&#10004;**            | **&#10004;**                |


## Individual Facial Detection and Recognition Results

| Image Name | Total Faces | Detected Faces | Available in Training Dataset | Correctly Identified Faces | Individual Image Accuracy |
|------------|-------------|----------------|-------------------------------|----------------------------|---------------------------|
| 1          | 5           | 5              | 3                             | 2                          | 66.67%                    |
| 2          | 8           | 8              | 6                             | 4                          | 66.67%                    |
| 3          | 5           | 5              | 5                             | 4                          | 80.00%                    |
| 4          | 4           | 4              | 3                             | 2                          | 66.67%                    |
| 5          | 7           | 7              | 5                             | 3                          | 60.00%                    |
| 6          | 5           | 5              | 2                             | 2                          | 100.00%                   |
| 8          | 4           | 4              | 2                             | 2                          | 100.00%                   |
| 10         | 12          | 11             | 8                             | 8                          | 100.00%                   |
| 14         | 14          | 14             | 12                            | 10                         | 83.33%                    |
| 15         | 9           | 9              | 9                             | 9                          | 100.00%                   |
| 16         | 11          | 11             | 10                            | 9                          | 90.00%                    |
| 17         | 9           | 9              | 8                             | 8                          | 100.00%                   |
| 18         | 7           | 7              | 7                             | 6                          | 85.71%                    |
| 19         | 5           | 5              | 4                             | 3                          | 75.00%                    |
| 20         | 10          | 10             | 10                            | 8                          | 80.00%                    |
| 21         | 5           | 5              | 5                             | 5                          | 100.00%                   |
| 22         | 5           | 5              | 4                             | 4                          | 100.00%                   |
| 23         | 4           | 4              | 4                             | 4                          | 100.00%                   |
| 24         | 3           | 3              | 2                             | 2                          | 100.00%                   |
| 26         | 2           | 2              | 2                             | 2                          | 100.00%                   |
| 27         | 7           | 7              | 6                             | 5                          | 83.33%                    |
| 28         | 4           | 4              | 4                             | 4                          | 100.00%                   |
| 29         | 4           | 4              | 4                             | 3                          | 75.00%                    |
| 30         | 12          | 12             | 9                             | 9                          | 100.00%                   |
| 32         | 6           | 6              | 6                             | 5                          | 83.33%                    |
| 33         | 3           | 3              | 2                             | 2                          | 100.00%                   |
| 35         | 3           | 3              | 1                             | 1                          | 100.00%                   |
| 36         | 5           | 5              | 2                             | 2                          | 100.00%                   |
| 37         | 5           | 5              | 1                             | 1                          | 100.00%                   |
| 38         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 39         | 10          | 10             | 4                             | 3                          | 75.00%                    |
| 40         | 9           | 9              | 5                             | 5                          | 100.00%                   |
| 41         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 42         | 12          | 12             | 6                             | 5                          | 83.33%                    |
| 43         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 44         | 11          | 11             | 5                             | 4                          | 80.00%                    |
| 45         | 11          | 11             | 5                             | 4                          | 80.00%                    |
| 46         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 47         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 48         | 11          | 11             | 5                             | 4                          | 80.00%                    |
| 49         | 11          | 11             | 5                             | 4                          | 80.00%                    |
| 50         | 11          | 11             | 6                             | 6                          | 100.00%                   |
| 51         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 52         | 11          | 11             | 5                             | 4                          | 80.00%                    |
| 53         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 54         | 13          | 13             | 6                             | 6                          | 100.00%                   |
| 55         | 11          | 11             | 5                             | 5                          | 100.00%                   |
| 56         | 10          | 10             | 5                             | 5                          | 100.00%                   |
| 57         | 27          | 26             | 18                            | 16                         | 88.89%                    |
| 58         | 13          | 12             | 9                             | 8                          | 88.89%                    |
| 59         | 12          | 12             | 5                             | 4                          | 80.00%                    |
| 60         | 14          | 14             | 5                             | 4                          | 80.00%                    |
| 61         | 21          | 21             | 10                            | 8                          | 80.00%                    |
| 62         | 26          | 25             | 12                            | 10                         | 83.33%                    |
| 63         | 16          | 16             | 8                             | 7                          | 87.50%                    |
| 64         | 18          | 18             | 8                             | 8                          | 100.00%                   |
| 65         | 20          | 20             | 11                            | 10                         | 90.91%                    |
| 66         | 5           | 5              | 4                             | 3                          | 75.00%                    |
| 67         | 5           | 5              | 4                             | 4                          | 100.00%                   |
| 68         | 5           | 5              | 4                             | 4                          | 100.00%                   |
| 69         | 2           | 2              | 1                             | 1                          | 100.00%                   |
|          |   Mean Detection Accuracy         |    99.4%           |                              | Mean Recognition Accuracy                          | 90.30%                   |


## Demo:
https://github.com/YoushanZhang/AiAI/assets/62828547/ca918506-9f5e-435f-9b19-e51496da88b8


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
1. [RetinaFace](https://arxiv.org/abs/1905.00641)
2. [Title Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.timedynamo.com%2Fblog%2Fface-recognition-attendance-system&psig=AOvVaw2FkF7iZtn_xnR0WqiLOgx8&ust=1713675455940000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMjdmdmA0IUDFQAAAAAdAAAAABAS)

