import cv2
import torch
import numpy as np
import mediapipe as mp
from week5_training import EmotionClassifier  # Make sure this matches your filename or adjust accordingly

# Load Label Mappings (manually copied from training script) 
index_to_label = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Load Trained Model 
model = EmotionClassifier(input_size=1404, hidden1=512, hidden2=256, hidden3=128, output_size=len(index_to_label))
# model = EmotionClassifier(input_size=1404, hidden_size=256, output_size=len(index_to_label))
model.load_state_dict(torch.load(r'C:\face-emoji\models\emotion_model.pth'))  # Same path from training
model.eval()

# Set Up Webcam & MediaPipe 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract and flatten landmark coords
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)

            if len(landmarks) == 1404:
                input_tensor = torch.tensor(landmarks).unsqueeze(0)  # shape: (1, 1404)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_class = torch.argmax(output, dim=1).item()
                    emotion = index_to_label[pred_class]

                # Get position for drawing text
                h, w, _ = frame.shape
                x = int(face_landmarks.landmark[0].x * w)
                y = int(face_landmarks.landmark[0].y * h)

                # Draw emotion label
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
