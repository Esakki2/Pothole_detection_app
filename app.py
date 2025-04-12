import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np
import requests
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# --- Initialize session state variables ---
if "detections" not in st.session_state:
    st.session_state.detections = []
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "api_status" not in st.session_state:
    st.session_state.api_status = "Unchecked"

st.title("🚧 Real-time Pothole Detection")

# --- API URL & Location Input ---
api_url = st.text_input("🔗 FastAPI Server URL", "https://your-api-url.ngrok-free.app")
latitude = st.number_input("📍 Latitude", value=0.0, format="%.6f")
longitude = st.number_input("📍 Longitude", value=0.0, format="%.6f")

# --- API Connectivity Check ---
def check_api_connection():
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "✅ Connected"
        else:
            st.session_state.api_status = f"⚠️ Error {response.status_code}"
    except Exception as e:
        st.session_state.api_status = f"❌ Failed: {e}"

if st.button("Check API Status"):
    check_api_connection()

st.markdown(f"**API Status:** {st.session_state.api_status}")

# --- Frame Processing Function ---
def process_frame(frame):
    st.session_state.frame_count += 1
    resized = cv2.resize(frame, (320, 320))
    success, encoded = cv2.imencode('.jpg', resized)
    if not success:
        return frame

    files = {"file": ("image.jpg", encoded.tobytes(), "image/jpeg")}
    data = {"latitude": latitude, "longitude": longitude}

    try:
        start_time = time.time()
        response = requests.post(f"{api_url}/process_frame/", files=files, data=data, timeout=10)
        elapsed = time.time() - start_time
        st.info(f"⏱️ Response Time: {elapsed:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])
        else:
            st.warning(f"⚠️ Server returned {response.status_code}")
            detections = []
    except Exception as e:
        st.error(f"❌ API error: {e}")
        detections = []

    # Draw bounding boxes
    scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 320
    pothole_found = False

    for det in detections:
        if det["confidence"] > 0.5:
            pothole_found = True
            x1 = int(det["x_min"] * scale_x)
            y1 = int(det["y_min"] * scale_y)
            x2 = int(det["x_max"] * scale_x)
            y2 = int(det["y_max"] * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if pothole_found:
        st.session_state.processed_frames.append((frame.copy(), detections.copy()))

    return frame

# --- WebRTC Video Processing ---
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

webrtc_streamer(
    key="pothole-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Display Stats ---
st.markdown(f"**📊 Total Frames Processed:** {st.session_state.frame_count}")
st.markdown(f"**🕳️ Pothole Detections:** {len(st.session_state.processed_frames)}")

# --- Show Latest Frame ---
if st.session_state.processed_frames:
    last_frame, last_detections = st.session_state.processed_frames[-1]
    st.image(last_frame, channels="BGR", caption=f"{len(last_detections)} potholes detected", width=400)

# --- PDF Report Generator ---
if st.button("📄 Generate PDF Report"):
    if st.session_state.processed_frames:
        pdf_file = "pothole_report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Pothole Detection Report")
        c.drawString(100, 730, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        y = 700

        for i, (frame, detections) in enumerate(st.session_state.processed_frames):
            img_path = f"frame_{i}.jpg"
            cv2.imwrite(img_path, frame)
            c.drawString(100, y, f"Frame {i+1}: {len(detections)} potholes detected")
            c.drawImage(img_path, 100, y - 150, width=400, height=300)
            y -= 170
            if y < 100:
                c.showPage()
                y = 750
            os.remove(img_path)

        c.save()
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download Report", f, file_name=pdf_file)
    else:
        st.warning("⚠️ No frames available to generate report.")
