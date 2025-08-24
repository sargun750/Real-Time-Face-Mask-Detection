import cv2
import numpy as np
import time
from keras.models import load_model

model = load_model("mask_model_final.h5")

deploy = "D:/Documents/vs cod/Face Mask/deploy.prototxt.txt"
model_path = "D:/Documents/vs cod/Face Mask/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(deploy, model_path)

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5: # min_confidence = 0.5
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
    return faces

def load_and_preprocess(img):
    
    img = cv2.resize(img, (224,224))  # Acc to Model's input
    img = img / 255.0                 # Normalize
    img = np.expand_dims(img, axis=0) # Add batch dimension

    return img

cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture")
        break
    
    new_frame_time = time.time()
    faces = detect_faces_dnn(frame) # Got the positioning of faces

    for (x, y, w, h, confidence) in faces:
        face = frame[y : y+h, x : x+w]
        norm_face = load_and_preprocess(face)

        pred = model.predict(norm_face)[0][0]
        label = "No Mask" if pred > 0.5 else "Mask Detected"
        color = (0, 0, 255) if pred > 0.5 else (0, 255, 0) # Green -> Detected, Red -> Not

        # Calculate FPS
        #latency = (new_frame_time*1000 - prev_frame_time*1000)
        prev_frame_time = new_frame_time

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} {confidence*100:.1f}%", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, \
                    1, color, 1)
        
    
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()