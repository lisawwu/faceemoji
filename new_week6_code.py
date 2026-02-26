import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained Keras emotion model
model_best = load_model(r'C:\face-emoji\models\face_model.h5')

# Class labels
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face detection
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Crop and preprocess the face region
            face_roi = frame[y:y + height, x:x + width]
            if face_roi.size == 0:
                continue  # skip if invalid region

            try:
                face_image = cv2.resize(face_roi, (48, 48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = image.img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_image = np.vstack([face_image])

                # Predict emotion
                predictions = model_best.predict(face_image, verbose=0)
                emotion_label = class_names[np.argmax(predictions)]

                # Annotate frame
                cv2.putText(frame, f'{emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

            except Exception as e:
                print("Face processing error:", e)

    # Display output
    cv2.imshow('Emotion Detection with MediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
