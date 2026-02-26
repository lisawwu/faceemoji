import os
import cv2 # load images
import csv #store data
import mediapipe as mp #extract landmarks
from tqdm import tqdm #optional progress bar

# only show 2 warning messages
os.environ['GLOG_minloglevel'] = '2'

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks(image, face_mesh):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #convert images to rgb
    if results.multi_face_landmarks: # if face found, extract 1404 landmark values
        return [coord
                for lm in results.multi_face_landmarks[0].landmark
                for coord in (lm.x, lm.y, lm.z)]
    return None

root_paths = [
    r'C:\face-emoji\images2\train',
    r'C:\face-emoji\images2\test',
    r'C:\face-emoji\images\train',
    r'C:\face-emoji\images\validation'
]#goes through both train and validation folders

output_file = r'C:\face-emoji\data\landmarks_dataset.csv' 
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh: 
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for root in root_paths: #loop through images
            for emotion in os.listdir(root):
                emotion_path = os.path.join(root, emotion)
                if not os.path.isdir(emotion_path):
                    continue

                print(f"Processing emotion: {emotion} in {root}")
                for img_file in tqdm(os.listdir(emotion_path)):
                    img_path = os.path.join(emotion_path, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    landmarks = extract_landmarks(image, face_mesh)
                    if landmarks:
                        writer.writerow(landmarks + [emotion]) #append emotion (label) at the end
