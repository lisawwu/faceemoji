import cv2
import mediapipe as mp

#initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection 
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5)

#initialize camera (week1 code)
cap = cv2.VideoCapture(0)
#check if camera is opened successfully 
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    #read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # WEEK 2 NEW CODE
    # convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process the frame and detect faces
    results = face_detection.process(rgb_frame)

    # draw face detections
    if results.detections:
        for detection in results.detections:
            # draw face detection box
            mp_drawing.draw_detection(frame, detection)

            # Get detection confidence and display it
            confidence = detection.score[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)

            # Display confidence score
            confidence_text = f'Confidence: {confidence:.2f}'
            cv2.putText(frame, confidence_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # (wk 1) display the video feed
    cv2.imshow('Video Feed', frame)

    #press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
face_detection.close()