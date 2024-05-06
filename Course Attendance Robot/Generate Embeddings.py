#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="text-align:center;">
#    <h1 style="font-size: 36px;">ClassVision: AI-Powered Classroom Attendance System</h1>
# </div>

# **Project Team:**
# - `Ankit Kumar Aggarwal`
# - `Veerabhadra Rao Marellapudi`
# - `Ovadia Sutton`

# ## Install Required Libraries

# In[ ]:


#Install Face Recognition Library
pip install face_recognition


# In[ ]:


#Install Retina Face
pip install retina-face


# ## Import Libraries

# In[ ]:


import os
import cv2
import pickle
import random
import face_recognition
import cv2
import pickle
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# ## Function to prepare Embeddings for Known Images

# In[ ]:


# Function to load or generate known face encodings and names
def load_or_generate_known_faces():
    # If the pickle file exists, load the known face encodings
    if os.path.exists("known_faces_5050px_a.pickle"):
        with open("known_faces_5050px_a.pickle", "rb") as file:
            known_faces = pickle.load(file)
        return known_faces

    # If the pickle file does not exist, generate the known face encodings and names
    known_faces_encodings = []
    known_face_names = []

    pbar = tqdm(desc="Processing Images")

    # Traverse through each image in the training directory
    training_directory = "/content/drive/MyDrive/AKA/04072024/training"
    for image_name in os.listdir(training_directory):
        if image_name.endswith((".jpg", ".jpeg")):
            image_path = os.path.join(training_directory, image_name)
            pbar.set_postfix({"Current Image": image_name})
            # print(f"Loading image: {image_path}")
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_faces_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(image_name)[0])
            else:
                # print(f"No faces detected in {image_name}. Skipping this image.")
                pass
            pbar.update(1)
    pbar.close()

    # Save the known face encodings and names to a pickle file
    known_faces = (known_faces_encodings, known_face_names)
    with open("known_faces_5050px_a.pickle", "wb") as file:
        pickle.dump(known_faces, file)

    print("Known faces encoding and names saved to 'known_faces_5050px_a.pickle'.")

    return known_faces

