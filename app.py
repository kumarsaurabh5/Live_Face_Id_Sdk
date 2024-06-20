from flask import Flask, render_template, Response, jsonify, redirect, url_for
import cv2
import dlib
import numpy as np
import face_recognition
import time
import mediapipe as mp
import os

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)  

def reset_video_capture():
    global video_capture
    if video_capture.isOpened():
        video_capture.release()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Could not restart video capture. Please ensure the webcam is connected.")

reset_video_capture()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

current_instruction = "Please blink your eyes."
status = "Awaiting action"
verification_status = None  
stop_video_feed = False 
blink_verified = False
hand_verified = False
head_movement_verified = False
captured_images = {'capture': None, 'liveliness': None}
percentage_match = None

def save_image(image, image_type):
    path = f'static/{image_type}.jpg'
    cv2.imwrite(path, image)
    captured_images[image_type] = path

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        return True
    return False

def detect_head_movement(frame, previous_landmarks, current_landmarks):
    movement_threshold = 20  
    if previous_landmarks is None:
        return False
    movement = np.linalg.norm(current_landmarks - previous_landmarks)
    return movement > movement_threshold

def gen_frames():
    global status, verification_status, stop_video_feed, blink_verified, hand_verified, head_movement_verified, current_instruction
    previous_landmarks = None

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture image")
            continue

        if frame is None:
            print("No frame captured")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if not blink_verified:
                if ear < 0.21:
                    blink_verified = True
                    current_instruction = "Please raise your right hand."
                break

            if blink_verified and not hand_verified:
                if detect_hand(frame):
                    hand_verified = True
                    current_instruction = "Please move your head."
                break

            if blink_verified and hand_verified and not head_movement_verified:
                if detect_head_movement(frame, previous_landmarks, landmarks):
                    head_movement_verified = True
                    current_instruction = "Liveliness check complete."
                previous_landmarks = landmarks
                break

        if blink_verified and hand_verified and head_movement_verified:
            status = "Live Person Detected"
            verification_status = True
            stop_video_feed = True
            cv2.putText(frame, "Verified", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            save_image(frame, 'liveliness')
            time.sleep(1)  # Wait for a moment to ensure the image is captured properly
            break
        elif stop_video_feed:
            status = "No Liveliness Detected"
            verification_status = False
            cv2.putText(frame, "Not Live", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global stop_video_feed, blink_verified, hand_verified, head_movement_verified
    stop_video_feed = False
    blink_verified = False
    hand_verified = False
    head_movement_verified = False
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', captured_images=captured_images, verification_status=verification_status, percentage_match=percentage_match)

@app.route('/reset', methods=['POST'])
def reset():
    global verification_status, status, stop_video_feed, blink_verified, hand_verified, head_movement_verified, captured_images, current_instruction, percentage_match
    verification_status = None
    status = "Awaiting action"
    stop_video_feed = False
    blink_verified = False
    hand_verified = False
    head_movement_verified = False
    captured_images = {'capture': None, 'liveliness': None}
    current_instruction = "Please blink your eyes."
    percentage_match = None
    reset_video_capture()
    return jsonify({"success": True})

@app.route('/liveliness_detection')
def liveliness_detection():
    return render_template('liveliness.html', current_instruction=current_instruction, status=status, verification_status=verification_status)

@app.route('/save_capture_image')
def save_capture_image():
    success, frame = video_capture.read()
    if success:
        save_image(frame, 'capture')
        return jsonify({"success": True, "image_url": captured_images['capture']})
    return jsonify({"success": False})

@app.route('/match_face')
def match_face():
    global verification_status, percentage_match
    if not captured_images['capture'] or not captured_images['liveliness']:
        verification_status = "Please upload the photos"
        return redirect(url_for('index'))

    img1 = face_recognition.load_image_file(captured_images['capture'])
    img2 = face_recognition.load_image_file(captured_images['liveliness'])

    try:
        img1_encoding = face_recognition.face_encodings(img1)[0]
        img2_encoding = face_recognition.face_encodings(img2)[0]

        matches = face_recognition.compare_faces([img1_encoding], img2_encoding)
        face_distance = face_recognition.face_distance([img1_encoding], img2_encoding)[0]
        percentage_match = round((1 - face_distance) * 100, 2)

        if matches[0]:
            verification_status = "Verified"
        else:
            verification_status = "Not Verified"
    except IndexError:
        if len(face_recognition.face_encodings(img1)) == 0:
            verification_status = "No face found in the captured image"
        elif len(face_recognition.face_encodings(img2)) == 0:
            verification_status = "No face found in the liveliness image"

    return redirect(url_for('index'))

@app.route('/check_verification_status')
def check_verification_status():
    global verification_status
    if verification_status is not None:
        return jsonify({"redirect": True, "url": url_for('index')})
    return jsonify({"redirect": False})

@app.route('/clear')
def clear():
    global verification_status, captured_images, percentage_match
    captured_images = {'capture': None, 'liveliness': None}
    verification_status = None
    percentage_match = None
    reset_video_capture()
    return redirect(url_for('index'))


