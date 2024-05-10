import os 
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from datetime import datetime, timedelta
import uuid
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
import sqlite3 
import logging
from PIL import Image
import io
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, PretrainedConfig
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import sounddevice as sd
from flask import abort, session
from werkzeug.utils import secure_filename

from werkzeug.security import check_password_hash
from flask_cors import CORS  # Import CORS

# from flask_bcrypt import Bcrypt

import sqlite3


import json

import whisper

# Suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged (default behavior)
                                          # 1 = INFO messages are not printed
                                          # 2 = INFO and WARNING messages are not printed
                                          # 3 = INFO, WARNING, and ERROR messages are not printed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_script_path():
    try:
        # This tries to get the full path of the current script
        full_path = os.path.realpath(__file__)
        return os.path.dirname(full_path)
    except NameError:
        # If __file__ is not defined, fallback to the current working directory
        return os.getcwd()

# Path to Current Working Directory
dir_path = get_script_path()
db_path = os.path.join(dir_path, 'users.db')
print(f"CWD: {dir_path}")
print(f"Database path: {db_path}")

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=0 if torch.cuda.is_available() else -1)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
# You can replace this embedding with your own as wel

stt_model = whisper.load_model("base")


def get_db_connection():
    # db_path = 'users.db'  # Ensure this is the correct and consistent path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    print(f"Connecting to the database")  # This line will confirm the DB path
    return conn

def does_table_exist(cursor, table_name):
    # Query to check if the specified table exists in the sqlite_master table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def setup_database():
    try:
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if the 'users' table exists and create it if it does not
        if not does_table_exist(cursor, 'users'):
            cursor.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, user_type TEXT)''')
            conn.commit()
            print("Users table created successfully")
        else:
            print("Users table already exists")

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
        print("Connection closed")

# Call the function to ensure the setup is processed
setup_database()

class VQAModel:
    def __init__(self):

        # Initialize the model with the loaded configuration
        self.model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-large", torch_dtype=torch.bfloat16).to(device)
        
        self.model_path = dir_path + '/models/model_weights_Pix2StructForConditionalGeneration__20_14 days, 3_52_26.pth'
        # print(f"Loading model from {self.model_path}")
        self.load_model()

        # Initialize the processor
        self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-large")

    # Load the model weights
    def load_model(self):

        # Load the state dict (the model weights)
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model: ", e)
        
        # Clear any cached memory to avoid memory leaks, especially useful when using CUDA
        torch.cuda.empty_cache()

        
    def predict(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(device, torch.bfloat16)
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=150)
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return answer

model = VQAModel()


app = Flask(__name__, static_folder='static')
app.logger.setLevel(logging.INFO)  # Set the logger level directly

# bcrypt = Bcrypt(app)  # Initialize the bcrypt extension

CORS(app)  # Initialize CORS on your app, this is crucial for cross-origin requests if needed


# Define the path for the audio folder within the user's home directory
AUDIO_FOLDER = os.path.join(app.root_path, 'temp_audio')

# Check if the directory exists, and if not, create it
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Function to clear the temp_audio folder on server start
def clear_audio_folder():
    for filename in os.listdir(AUDIO_FOLDER):
        file_path = os.path.join(AUDIO_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f'Cleared {filename} from audio folder.')
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Call the function to clear the audio folder on server start
clear_audio_folder()

# Function to clean up old files periodically
def cleanup_old_files():
    now = datetime.now()
    for filename in os.listdir(AUDIO_FOLDER):
        file_path = os.path.join(AUDIO_FOLDER, filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if now - file_time > timedelta(hours=1):  # Adjust time as needed
            os.remove(file_path)
            print(f"Deleted {filename} due to expiry")

# Scheduler to clean up files every hour
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_files, trigger="interval", hours=1)
scheduler.start()

# Shutdown the scheduler when the web process is stopped
atexit.register(lambda: scheduler.shutdown())

@app.route('/get-audio/<filename>')
def get_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/api/text2speech', methods=['POST'])
def text2speech():
    max_length = 598  # Adjust based on your model's specific limitations

    text = request.json['text']
    print(text)

    if len(text) > max_length:
        print(f"Text length ({len(text)}) exceeds the maximum allowed length ({max_length}). Truncating...")
        text = text[:max_length]

    try:
        speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
        
        # Save the audio to a file
        audio_filename = f"{uuid.uuid4()}.wav"  # Generate a unique filename
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        sf.write(audio_path, speech["audio"], speech["sampling_rate"])
        
        # Generate a URL to access the audio file through the get_audio route
        audio_url = url_for('get_audio', filename=audio_filename, _external=True)
        
        return jsonify({"status": "success", "message": "Audio generated successfully", "audioUrl": audio_url})
    except Exception as e:
        print(f"Error in text-to-speech generation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        question = request.form['question']
        answer = model.predict(image, question)
        return jsonify({'answer': answer})
    

# Set a course into session after user selects it
@app.route('/set_course/<course_name>')
def set_course(course_name):
    session['current_course'] = course_name
    return 'Course set to ' + course_name

@app.route('/course/<course>/<week>/<filename>')
def course_material(course, week, filename):
    path = f"courses/{course}/{week}"
    return send_from_directory(app.static_folder, f"{path}/{filename}")

@app.route('/api/courses')
def get_courses():
    courses_path = os.path.join(app.static_folder, "courses")
    default_image = 'course_image.webp'  # Default image name
    courses = []
    # Scan the directory for courses
    for course in os.listdir(courses_path):
        course_dir = os.path.join(courses_path, course)
        if os.path.isdir(course_dir):
            # Path for course-specific image
            course_image_path = os.path.join(course_dir, default_image)
            # Check if the course-specific image exists, otherwise use the default
            if not os.path.exists(course_image_path):
                course_image_path = os.path.join(app.static_folder, 'courses', default_image)
                image_url = url_for('static', filename=f'courses/{default_image}')
            else:
                image_url = url_for('static', filename=f'courses/{course}/{default_image}')
            
            courses.append({
                'name': course.replace("_", " "),  # Convert directory name to course title
                'image': image_url
            })
    return jsonify(courses)

@app.route('/materials/<path:filename>')
def get_material(filename):
    # Secure the filename to prevent directory traversal
    filename = secure_filename(filename)
    filepath = os.path.join(app.static_folder, filename)
    if os.path.exists(filepath):
        return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))
    abort(404, "Resource not found")

@app.route('/api/lectures/<course_name>')
def lectures_for_course(course_name):
    base_path = os.path.join(app.static_folder, "courses", course_name, "lectures")
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        return jsonify([])  # If no directory, return empty list

    lectures = []
    try:
        # List all directories within the course lectures path
        for week_dir in sorted(os.listdir(base_path)):
            week_path = os.path.join(base_path, week_dir)
            if os.path.isdir(week_path):
                images = sorted(os.listdir(week_path))
                if images:
                    first_image = images[0]
                    lectures.append({
                        'week': week_dir,
                        'img': url_for('static', filename=f'courses/{course_name}/lectures/{week_dir}/{first_image}')
                    })
        return jsonify(lectures)
    except Exception as e:
        app.logger.error(f"Failed to fetch lectures: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

# @app.route('/api/lecture-images/<course_name>/lectures/<week>')
@app.route('/api/lecture-images/<course_name>/<week>')
def lecture_images(course_name, week):
    app.logger.info("Fetching lecture image paths...")

    # Building the path to the directory containing lecture images
    lecture_path = os.path.join(app.static_folder, "courses", course_name, "lectures", week)
    app.logger.info("course_name: %s", course_name)
    app.logger.info("week: %s", week)
    app.logger.info("lecture_path: %s", lecture_path)
    
    if not os.path.exists(lecture_path) or not os.path.isdir(lecture_path):
        app.logger.error(f"No directory found for {lecture_path}")
        return jsonify([])  # Return an empty list if the directory doesn't exist

    # images = []
    image_paths = []  # Initialize an empty list to store image paths
    try:
        # List all image files in the directory
        for img_file in sorted(os.listdir(lecture_path)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):  # Filter for image files
                # img_url = url_for('static', filename=f'courses/{course_name}/lectures/{week}/{img_file}')
                # image_paths.append(img_url)
                image_paths.append(f'/static/courses/{course_name}/lectures/{week}/{img_file}')

        # app.logger.info("Image paths: %s", image_paths)
        return jsonify(image_paths)
    except Exception as e:
        app.logger.error(f"Failed to load images for {course_name}/{week}: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

    
@app.route('/api/vqa', methods=['POST'])
def handle_vqa():
    try:
        # Extract the image and question from the request
        file = request.files['image']
        question = request.form['question']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Get the answer from the model
        answer = model.predict(image, question)

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in VQA processing: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return "Audio file is required.", 400
    
    audio_file = request.files['audio']
    audio_path = "./temp_audio.ogg"  # Temporary save location
    audio_file.save(audio_path)
    
    # Transcribe the audio
    result = stt_model.transcribe(audio_path)
    transcription = result["text"]

    # Cleanup and response
    os.remove(audio_path)  # Clean up the temporary file
    return jsonify({"tra nscription": transcription})

#create user
@app.route('/user', methods=['POST'])
def create_user():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)',
                   (data['username'], data['password'], data['user_type']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'User created successfully'}), 201

#update user
@app.route('/user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET username = ?, password = ?, user_type = ? WHERE id = ?',
                   (data['username'], data['password'], data['user_type'], user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'User updated successfully'})

#delete user
@app.route('/user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'User deleted successfully'})
    
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print("Received data:", data)  # Debugging line to check what is received

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'status': 'Error', 'message': 'Missing username or password'}), 400

    username = data['username']
    password = data['password']
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query to fetch the user details
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        # If user exists, handle login success
        return jsonify({
            'status': 'Logged in successfully',
            'user_type': user['user_type']
        })
    else:
        # If no user found, handle login failure
        return jsonify({'status': 'Login failed'}), 401

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return jsonify({'status': 'Logged out successfully'})

#get all users
@app.route('/users/all', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

#get all admins
@app.route('/users/admins', methods=['GET'])
def get_admins():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_type = 'admin'")
    admins = cursor.fetchall()
    conn.close()
    return jsonify([dict(admin) for admin in admins])

#get all users
@app.route('/users/regular', methods=['GET'])
def get_regular_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_type = 'user'")
    regular_users = cursor.fetchall()
    conn.close()
    return jsonify([dict(user) for user in regular_users])

@app.route('/user_administration')
def user_administration():
    return render_template('user_administration.html')

if __name__ == '__main__':
    app.run(debug=True)
