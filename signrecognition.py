import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the camera (change the index if needed)
cap = cv2.VideoCapture(0)  # Try different indices if necessary

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'hello', 
               11: 'yes', 12: 'no', 13: 'good', 14: 'good afternoon', 15: 'good night', 16: 'thank you', 
               17: 'i love you', 18: 'please', 19: 'A', 20: 'B', 21: 'C', 22: 'D', 23: 'E', 24: 'F', 
               25: 'G', 26: 'H', 27: 'I', 28: 'J', 29: 'K', 30: 'L', 31: 'M', 32: 'N', 33: 'O', 34: 'P', 
               35: 'Q', 36: 'R', 37: 'S', 38: 'T', 39: 'U', 40: 'V', 41: 'W', 42: 'X', 43: 'Y', 44: 'Z'}

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break  # Exit loop if frame capture fails

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        # Draw hand landmarks and collect coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Calculate data_aux for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

        # If only one hand is detected, duplicate the features to match the model's expected input (84 features)
        if len(results.multi_hand_landmarks) == 1:
            data_aux = data_aux + data_aux  # Duplicate the feature vector for two hands

        # Ensure the feature vector has 84 values (42 features per hand, 2 hands)
        elif len(results.multi_hand_landmarks) > 1:
            # If two hands are detected, just ensure that both hands' landmarks are processed
            # This part might already be handled in the loop above, but ensure the right feature vector size

            pass  # No need for padding if both hands are detected

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()