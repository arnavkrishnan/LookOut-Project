import streamlit as st
import cv2
from ultralytics import YOLO
import threading
import time
import pyttsx3
import numpy as np

# Initialize Text-to-Speech (TTS) engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech rate

# Function to speak text asynchronously
def speak_async(text):
    def speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=speak).start()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the Nano model for speed

# Streamlit app setup
st.set_page_config(page_title="LookOut - Captions & Audio Detection", layout="wide")
st.title("👁️ LookOut: See & Hear What's Around You")

# Create UI elements
run_detection = st.button("🔴 Start Camera")
frame_placeholder = st.image([])      # Placeholder for webcam feed
caption_placeholder = st.empty()      # Placeholder for captions

# Open webcam capture
cap = cv2.VideoCapture(0)

# List to hold the last 3 detection captions
recent_detections = []
last_speak_time = time.time()

if run_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Use the frame at its original resolution
        original_frame = frame

        # Run YOLOv8 detection on the current frame
        results = model(original_frame, verbose=False)[0]

        # Draw bounding boxes and labels on the frame
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            if conf > 0.5:
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update the webcam frame in the UI
        frame_placeholder.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))

        # Every 3 seconds, process detections and update captions
        current_time = time.time()
        if current_time - last_speak_time > 3:
            object_positions = []
            # For each detected box, determine the position (left, in front, or right)
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                # Determine spatial direction based on the center x coordinate
                center_x = (x1 + x2) / 2
                width = original_frame.shape[1]
                if center_x < width / 3:
                    direction = "left"
                elif center_x < 2 * width / 3:
                    direction = "in front"
                else:
                    direction = "right"

                object_positions.append(f"{label} to your {direction}")

            # If any objects were detected, update captions and speak aloud
            if object_positions:
                description = ", ".join(object_positions)

                # Append the new detection and keep only the last three
                recent_detections.append(description)
                if len(recent_detections) > 3:
                    recent_detections.pop(0)

                # Update the caption placeholder with the last three detections
                caption_placeholder.text("\n".join(recent_detections))

                # Speak the new detection aloud
                speak_async("I see " + description)

            last_speak_time = current_time

    cap.release()  # Release the webcam when done

else:
    st.info("Press the button to start the camera and object detection.")
