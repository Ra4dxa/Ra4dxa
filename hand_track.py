import cv2
import mediapipe as mp
import time
import os
import numpy as np  # ✅ Added this

# Initialize camera and mediapipe
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

pTime = 0
current_slide = 0
last_action_time = 0

# Load your slides (make sure these exist in the same folder)
slides = ["slide1.png", "slide2.png", "slide3.png"]

def count_fingers(hand_landmarks, img):
    """Count fingers based on landmark positions"""
    h, w, _ = img.shape
    lm = []
    for landmark in hand_landmarks.landmark:
        lm.append((int(landmark.x * w), int(landmark.y * h)))

    fingers = []
    tipIds = [4, 8, 12, 16, 20]

    # Thumb (left/right based on camera view)
    if lm[tipIds[0]][0] < lm[tipIds[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for id in range(1, 5):
        if lm[tipIds[id]][1] < lm[tipIds[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera not found or cannot read frame.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            totalFingers = count_fingers(handLms, img)
            current_time = time.time()

            # Gesture logic: 5 fingers → next slide, 0 fingers → previous
            if totalFingers == 5 and current_time - last_action_time > 1:
                current_slide = (current_slide + 1) % len(slides)
                last_action_time = current_time
                print(f"➡️ Next Slide: {current_slide + 1}")

            elif totalFingers == 0 and current_time - last_action_time > 1:
                current_slide = (current_slide - 1) % len(slides)
                last_action_time = current_time
                print(f"⬅️ Previous Slide: {current_slide + 1}")

    # Display the current slide
    slide_path = slides[current_slide]
    if not os.path.exists(slide_path):
        print(f"⚠️ Slide '{slide_path}' not found. Showing placeholder.")
        slide = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(slide, "Slide Not Found", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        slide = cv2.imread(slide_path)
        if slide is not None:
            slide = cv2.resize(slide, (640, 480))
        else:
            slide = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(slide, "Error Loading Slide", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Slide", slide)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
