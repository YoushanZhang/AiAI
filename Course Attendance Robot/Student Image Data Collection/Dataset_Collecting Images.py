### How to run the code ###

# 1. Create a folder Face_Recognition_Images
# 2. Copy "haarcascade_frontalface_default.xml" attached in the email into Face_Recognition_Images folder
# 3. Run the code 
# 4. Finally email the collected folder to vmarella@mail.yu.edu 

# Install these libraries if not installed in your system
# --- pip install opencv-python


import cv2
import os
import time

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('C:/Users/vijay/OneDrive/Desktop/Student Image Data Collection/haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Ask for the student's name
student_name = input('\nEnter student name and press <return> ==>  ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

# Automatically create a folder based on student_name
save_folder = f"C:/Users/vijay/OneDrive/Desktop/Student Image Data Collection/{student_name}/"
os.makedirs(save_folder, exist_ok=True)

# Initialize individual sampling face count
count = 0

while True:
    ret, frame = cap.read()

    if face_extractor(frame) is not None:
        count += 1
        # Resize the face to match 720p resolution
        face = cv2.resize(face_extractor(frame), (112, 112))

        img_name = f"{student_name}_{count}.jpg"
        img_path = os.path.join(save_folder, img_name)
        cv2.imwrite(img_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

        # Introduce a delay of 120 milliseconds between captures
        time.sleep(0.12)

    else:
        print("Face found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Completed Collecting Image Samples")

