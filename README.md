FACE RECOGNITION :

Real-Time Object Classification using Teachable Machine
This project demonstrates how to create a real-time object classification system using a webcam and a model trained with Google Teachable Machine. The Python script uses TensorFlow, OpenCV, and NumPy to classify objects and display predictions with confidence scores.

Requirements
Python 3.x
TensorFlow (For loading and using the pre-trained model)
OpenCV (For capturing real-time video from the webcam)
NumPy (For processing images before feeding them to the model)
Dependencies
You can install the necessary libraries by running:

bash
Copy code
pip install tensorflow opencv-python numpy
Project Structure
The project directory should look like this:

bash
Copy code
├── keras_Model.h5             # Trained model file from Teachable Machine
├── labels.txt                 # File containing class labels for the model
├── real_time_classification.py  # Python script for real-time classification

1. Model File (keras_Model.h5)
This is the trained model you exported from Google Teachable Machine. You will use this file to classify objects in real-time.

3. Labels File (labels.txt)
This file contains the labels (class names) corresponding to the output of your model. Each line should represent a class name.

Setup Instructions
Step 1: Train the Model using Google Teachable Machine
Go to Teachable Machine.
![image](https://github.com/user-attachments/assets/7680f44d-0e83-41ce-a9e6-97787d5f1854)

Create a new Image Project.
![image](https://github.com/user-attachments/assets/f7e7d656-4872-4419-a458-6d3b64028caf)
![image](https://github.com/user-attachments/assets/ab5b2c96-19df-452a-9497-d0017b9657a0)

Add the classes you want the model to recognize.

Train the model with your images (you can use your webcam or upload pictures).
![image](https://github.com/user-attachments/assets/d856308c-0159-4683-87f7-4bd92f0e903d)

After training, export the model as Keras Model (.h5).
Download the labels.txt file, which contains the class names corresponding to your model.
Step 2: Install Required Libraries
Ensure you have Python installed, and use the following command to install dependencies:

bash
Copy code
pip install tensorflow opencv-python numpy
Step 3: Set up Your Project Directory
Place the following files in your project folder:

keras_Model.h5 (the model you downloaded from Teachable Machine)
labels.txt (the class names for your model)
real_time_classification.py (the Python script to run the classification)
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels, stripping any newline characters
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    # Check if frame was captured correctly
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    confidence_score = np.round(confidence_score * 100, 2)
    print(f"Class: {class_name}, Confidence Score: {confidence_score}%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()


Usage
Run the Python script:

bash
Copy code
python real_time_classification.py
Real-Time Object Classification:

The webcam feed will open and the system will classify objects based on the model's predictions.
It will display the predicted class and the confidence score in the terminal.
Stopping the Program:

Press the ESC key to stop the program.
Code Explanation
1. Load the Model
python
Copy code
model = load_model("keras_Model.h5", compile=False)
This loads the pre-trained Keras model (.h5 file).

2. Load the Labels
python
Copy code
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
This reads the labels.txt file and loads the class names into a list.

3. Capture Webcam Feed
python
Copy code
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
This initializes the webcam (camera 0 is the default).

4. Preprocess the Image
The image captured from the webcam is resized and normalized before passing it to the model:

python
Copy code
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
image = (image / 127.5) - 1
5. Predict the Class
python
Copy code
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
This code predicts the class of the image and retrieves the confidence score.

6. Display Results
The predicted class and confidence score are printed in the terminal:
![image](https://github.com/user-attachments/assets/bd72a017-be77-4996-8f82-b3b5cad76e98)


python
Copy code
confidence_score = np.round(confidence_score * 100, 2)
print(f"Class: {class_name}, Confidence Score: {confidence_score}%")
![image](https://github.com/user-attachments/assets/e69f19b9-673e-4a4b-92e9-24890fb2745d)

7. Exit on ESC
python
Copy code
if keyboard_input == 27:  # 27 is the ASCII code for the ESC key
    break
The program listens for the ESC key to stop the loop and close the webcam.

Troubleshooting
Webcam not detected: Make sure your camera is connected and properly configured. Try using a different camera index like cv2.VideoCapture(1).
Model loading issues: Ensure that keras_Model.h5 is in the correct directory and is properly trained.
No class prediction: Make sure the model is trained with the correct number of classes and that labels.txt matches the model's class output.
License
This project is open-source. You are free to use, modify, and distribute it as per your needs.

