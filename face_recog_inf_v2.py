# python 3.8

import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from modules.face_recog_v2 import get_encodings_from_face_array

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_lst = []

distance_threshold=0.6


# Load a trained KNN model (if one was passed in)

with open("models/trained_knn_model_v2.clf", 'rb') as f:
    knn_clf = pickle.load(f)

# Create a face detector instance
detector = MTCNN()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = detector.detect_faces(rgb_small_frame)
    #     print(face_locations)
        face_locations_boxes = []
        face_names = []
        for idx, face_loc in enumerate(face_locations):
            
            x1, y1, width, height = face_loc['box']
            x2, y2 = x1 + width, y1 + height
            face_locations_boxes.append((y1, x1, y2, x2))
            # plot face if draw_faces is True
            cropped_face = rgb_small_frame[y1:y2, x1:x2]
            
            encoding = get_encodings_from_face_array(cropped_face)
            label = knn_clf.predict(encoding.reshape(1, -1))[0]

            face_names.append(label)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, left, bottom, right), name in zip(face_locations_boxes, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()