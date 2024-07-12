# Face Verification System

## Overview
The Face Verification System is a web application designed to ensure user identity verification through a combination of liveliness detection and face comparison. Utilizing a webcam, the system performs a series of checks to confirm that the user is physically present and not using a static image or video. The application is built using Flask for the backend, and leverages OpenCV, dlib, face_recognition, and Mediapipe libraries for various computer vision tasks.

## Features
1. **Capture Image**: Capture an image using the webcam for reference.
2. **Liveliness Detection**: Verify the presence of a live person through specific actions: eye blinking, hand raising, and head movement.
3. **Face Verification**: Compare the captured image with a liveliness-detected image to ensure they are of the same person.
4. **Reset Functionality**: Reset the application to clear the captured images and restart the verification process.

## Technologies Used
- **Flask**: A lightweight WSGI web application framework for Python.
- **OpenCV**: An open-source computer vision and machine learning software library.
- **dlib**: A toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real-world problems.
- **face_recognition**: A library built on top of dlib’s facial recognition capabilities.
- **Mediapipe**: A framework for building multimodal (e.g., video, audio, and sensor data) applied machine learning pipelines.

## Getting Started

### Prerequisites
- Python 3.x
- Pip (Python package installer)
- A webcam

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kumarsaurabh5/Live_Face_Id_Sdk-verification-system.git
   cd face-verification-system
   ```

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the shape predictor model**:
   Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

### Running the Application

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to `http://127.0.0.1:5000/` to access the Face Verification System.

## Usage

1. **Capture Image**: Click the "Capture Image" button to capture an image using the webcam. This image will be used as the reference image for face verification.
2. **Liveliness Detection**: Click the "Liveliness Detection" button to initiate the liveliness detection process. Follow the on-screen instructions to blink your eyes, raise your right hand, and move your head. This ensures the person in front of the camera is live and present.
3. **Match Face**: Click the "Match Face" button to compare the captured image with the liveliness-detected image. The system will display the verification status and the percentage match between the two images.
4. **Clear**: Click the "Clear" button to reset the application, clear the captured images, and start a new verification process.

## Project Structure

```
.
├── app.py                  # Main Flask application
├── templates
│   ├── index.html          # Home page template
│   └── liveliness.html     # Liveliness detection page template
├── static
│   ├── capture.jpg         # Placeholder for captured image
│   └── liveliness.jpg      # Placeholder for liveliness image
├── requirements.txt        # List of required Python packages
└── README.md               # This file
```

## Detailed Workflow

1. **Image Capture**:
   - The user clicks the "Capture Image" button.
   - The webcam captures an image and saves it as a reference image for face verification.

2. **Liveliness Detection**:
   - The user initiates the liveliness detection process.
   - The system displays instructions to perform specific actions:
     - Blink eyes
     - Raise right hand
     - Move head
   - The system uses dlib to detect facial landmarks and verify eye blinking.
   - Mediapipe is used to detect hand raising.
   - Head movement is detected by comparing the positions of facial landmarks over time.

3. **Face Verification**:
   - The captured image and liveliness-detected image are compared using the face_recognition library.
   - The system calculates the face distance and determines the percentage match.
   - A match percentage above a certain threshold indicates successful verification.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [dlib](http://dlib.net/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://opencv.org/)
- [Mediapipe](https://mediapipe.dev/)


This README provides an extended overview of the project, installation instructions, detailed usage guide, and information about the technologies used. It helps users understand the purpose of the application and how to set it up and run it locally, along with a detailed explanation of the workflow.
