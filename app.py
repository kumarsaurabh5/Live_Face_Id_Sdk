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
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return True
    return False

def detect_head_movement(previous_landmarks, current_landmarks):
    previous_points = np.array([[p.x, p.y] for p in previous_landmarks.parts()])
    current_points = np.array([[p.x, p.y] for p in current_landmarks.parts()])
    displacement = np.linalg.norm(current_points - previous_points, axis=1).mean()
    return displacement > 0.01

@app.route('/')
def index():
    global captured_images, verification_status, percentage_match
    return render_template('index.html', captured_images=captured_images, verification_status=verification_status, percentage_match=percentage_match)

@app.route('/save_capture_image')
def save_capture_image():
    global captured_images
    success, frame = video_capture.read()
    if success:
        save_image(frame, 'capture')
        return jsonify({'success': True, 'image_url': captured_images['capture']})
    return jsonify({'success': False})

@app.route('/liveliness_detection')
def liveliness_detection():
    global current_instruction, status
    current_instruction = "Please blink your eyes."
    status = "Awaiting action"
    return render_template('liveliness.html', current_instruction=current_instruction, status=status)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    global current_instruction, status, stop_video_feed, blink_verified, hand_verified, head_movement_verified, captured_images, video_capture

    blink_threshold = 0.25
    consecutive_frames = 3
    blink_counter = 0

    previous_landmarks = None
    start_time = time.time()

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                 (landmarks.part(37).x, landmarks.part(37).y),
                                 (landmarks.part(38).x, landmarks.part(38).y),
                                 (landmarks.part(39).x, landmarks.part(39).y),
                                 (landmarks.part(40).x, landmarks.part(40).y),
                                 (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                  (landmarks.part(43).x, landmarks.part(43).y),
                                  (landmarks.part(44).x, landmarks.part(44).y),
                                  (landmarks.part(45).x, landmarks.part(45).y),
                                  (landmarks.part(46).x, landmarks.part(46).y),
                                  (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < blink_threshold:
                blink_counter += 1
            else:
                if blink_counter >= consecutive_frames:
                    blink_verified = True
                blink_counter = 0

            if current_instruction == "Please blink your eyes." and blink_verified:
                current_instruction = "Please raise your right hand."
                status = "Awaiting action"

            if current_instruction == "Please raise your right hand." and detect_hand(frame):
                hand_verified = True

            if current_instruction == "Please raise your right hand." and hand_verified:
                current_instruction = "Please move your head."
                status = "Awaiting action"
                previous_landmarks = landmarks

            if current_instruction == "Please move your head." and previous_landmarks:
                if detect_head_movement(previous_landmarks, landmarks):
                    head_movement_verified = True

            if current_instruction == "Please move your head." and head_movement_verified:
                stop_video_feed = True
                save_image(frame, 'liveliness')
                status = "Liveliness Verified"
                return redirect('/')

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if stop_video_feed:
            break

@app.route('/match_face')
def match_face():
    global captured_images, verification_status, percentage_match

    if captured_images['capture'] and captured_images['liveliness']:
        capture_image = face_recognition.load_image_file(captured_images['capture'])
        liveliness_image = face_recognition.load_image_file(captured_images['liveliness'])

        capture_encoding = face_recognition.face_encodings(capture_image)[0]
        liveliness_encoding = face_recognition.face_encodings(liveliness_image)[0]

        distance = face_recognition.face_distance([capture_encoding], liveliness_encoding)[0]
        print(distance)
        
        percentage_match = (1 - distance) * 100
        print(percentage_match)
    

        if percentage_match > 50:
            verification_status = "Verification Successful"
        else:
            verification_status = "Verification Failed"
    else:
        verification_status = "Images not available for comparison"

    return redirect('/')

@app.route('/clear')
def clear():
    global captured_images, verification_status, percentage_match, stop_video_feed, blink_verified, hand_verified, head_movement_verified
    captured_images = {'capture': None, 'liveliness': None}
    verification_status = None
    percentage_match = None
    stop_video_feed = False
    blink_verified = False
    hand_verified = False
    head_movement_verified = False
    reset_video_capture()
    return redirect('/')

@app.route('/reset', methods=['POST'])
def reset():
    global current_instruction, status, stop_video_feed, blink_verified, hand_verified, head_movement_verified
    stop_video_feed = False
    blink_verified = False
    hand_verified = False
    head_movement_verified = False
    current_instruction = "Please blink your eyes."
    status = "Awaiting action"
    return jsonify({'success': True})

@app.route('/check_verification_status')
def check_verification_status():
    global stop_video_feed
    if stop_video_feed:
        return jsonify({'redirect': True, 'url': url_for('index')})
    return jsonify({'redirect': False})

if __name__ == '__main__':
    app.run(debug=True)
