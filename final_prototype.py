import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load model
model_path = "251ep-NOZ-INDEX-BEST.keras"
model = keras.models.load_model(model_path)

# Recognized gestures
actions = np.array([
    'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING', 'HELLO', 
    'HOW ARE YOU', 'IM FINE', 'NICE TO MEET YOU', 'THANK YOU', 
    'YOURE WELCOME', 'SEE YOU TOMORROW', 'MONDAY', 'TUESDAY', 
    'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 
    'TODAY', 'TOMORROW', 'YESTERDAY', 'BLUE', 'GREEN', 'RED', 
    'BROWN', 'BLACK', 'WHITE', 'YELLOW', 'ORANGE', 'GRAY', 'PINK', 
    'VIOLET', 'LIGHT', 'DARK'
])


def mediapipe_detection(image, model):
    """Processes the image and returns the detected results."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    """Draws the detected landmarks on the image."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    """Extracts keypoints from the results."""
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)
    return np.concatenate([pose, lh, rh])

# Setup video capture
cap = cv2.VideoCapture(0)
sequence = []
recognized_action = None

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-120:]  # Keep the last 120 keypoints

        if len(sequence) == 120:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            recognized_action = actions[np.argmax(res)]
        
        # Display recognized action at the top of the image
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, recognized_action if recognized_action else '...', (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
