import streamlit as st
import cv2
import numpy as np
import requests
import time
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ------------------- Session State Initialization ------------------- #
default_states = {
    "streaming": False,
    "frame_count": 0,
    "detections": [],
    "latitude": None,
    "longitude": None,
    "processed_frames": []
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------- Streamlit UI Setup ------------------- #
st.title("🚧 Real-time Pothole Detection")

api_url = st.text_input("🔗 Enter Server API URL", 
                        "https://1fd0-2402-3a80-4273-e6c3-f0df-5dc6-4ca5-5b58.ngrok-free.app")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Video Stream"):
        st.session_state.streaming = True
with col2:
    if st.button("⏹️ Stop Video Stream"):
        st.session_state.streaming = False
with col3:
    if st.button("📄 Generate PDF"):
        def generate_pdf_report():
            if not st.session_state.processed_frames:
                st.warning("No pothole detections available to include in the PDF.")
                return

            pdf_path = "pothole_detection_report.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(100, 750, "Pothole Detection Report")
            c.drawString(100, 735, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            y = 700
            for idx, (frame, detections) in enumerate(st.session_state.processed_frames):
                img_name = f"frame_{idx}.jpg"
                cv2.imwrite(img_name, frame)
                c.drawString(100, y, f"Frame {idx + 1}: {len(detections)} potholes detected")
                c.drawImage(img_name, 100, y - 150, width=400, height=300)
                y -= 170

                if y < 100:
                    c.showPage()
                    y = 750

                os.remove(img_name)

            c.save()
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("⬇️ Download PDF", pdf_file, file_name=pdf_path)

        generate_pdf_report()

# ------------------- Frame Processing Function ------------------- #
def process_frame(frame):
    st.session_state.frame_count += 1

    resized = cv2.resize(frame, (320, 320))
    success, encoded_img = cv2.imencode('.jpg', resized)
    if not success:
        return frame

    try:
        files = {"file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")}
        data = {
            "latitude": st.session_state.latitude,
            "longitude": st.session_state.longitude
        }
        response = requests.post(f"{api_url}/process_frame/", files=files, data=data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])
            st.session_state.detections = detections

            pothole_found = False
            scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 320

            for det in detections:
                if det["confidence"] > 0.5:
                    pothole_found = True
                    x1 = int(det["x_min"] * scale_x)
                    y1 = int(det["y_min"] * scale_y)
                    x2 = int(det["x_max"] * scale_x)
                    y2 = int(det["y_max"] * scale_y)
                    label = f"{det['class_name']} {det['confidence']:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if pothole_found:
                st.session_state.processed_frames.append((frame.copy(), detections))

    except Exception as e:
        st.error(f"⚠️ Error while processing frame: {e}")

    return frame

# ------------------- Live Video Stream Handler ------------------- #
def run_video_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Unable to access the camera.")
        return

    st.subheader("📹 Live Camera Feed")
    video_container = st.empty()

    st.subheader("✅ Processed Frames")
    processed_container = st.empty()

    while st.session_state.streaming:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera read failed.")
            break

        processed = process_frame(frame)

        video_container.image(processed, channels="BGR", caption=f"Frame {st.session_state.frame_count}")

        if st.session_state.processed_frames:
            recent_frame, recent_detections = st.session_state.processed_frames[-1]
            processed_container.image(
                recent_frame,
                channels="BGR",
                caption=f"Potholes Detected: {len(recent_detections)}",
                width=300
            )

        time.sleep(0.1)

    cap.release()

# ------------------- Stream Trigger ------------------- #
if st.session_state.streaming:
    run_video_stream()
else:
    st.info("Click 'Start Video Stream' to begin real-time pothole detection.")
