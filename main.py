import face_recognition
import cv2
import numpy as np
import os

def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return face_recognition.load_image_file(file_path)

def main():
    video_capture = cv2.VideoCapture(0)

    try:
        saurabh_image = load_image("Saurabh/saurabh.jpg")
        saurabh_face_encoding = face_recognition.face_encodings(saurabh_image)[0]

        abhishek_image = load_image("Abhishek/abhishek.jpg")
        abhishek_face_encoding = face_recognition.face_encodings(abhishek_image)[0]
    except FileNotFoundError as e:
        print(e)
        return
    except IndexError:
        print("No face found in the image(s).")
        return

    known_face_encodings = [saurabh_face_encoding, abhishek_face_encoding]
    known_face_names = ["saurabh", "abhishek"]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        
        if process_this_frame:
        
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
