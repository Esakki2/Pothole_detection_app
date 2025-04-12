import streamlit as st
import cv2
import numpy as np
import requests
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Initialize session state variables
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "detections" not in st.session_state:
    st.session_state.detections = []
if "latitude" not in st.session_state:
    st.session_state.latitude = None
if "longitude" not in st.session_state:
    st.session_state.longitude = None
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = []  # Store only frames with potholes

# Streamlit page configuration
st.set_page_config(page_title="Pothole Detection App", layout="wide")

st.title("Pothole Detection App")
st.write("Stream live video from your webcam to detect potholes using your browser's camera.")

# Input for server API URL
api_url = st.text_input("Server API URL", "https://1fd0-2402-3a80-4273-e6c3-f0df-5dc6-4ca5-5b58.ngrok-free.app")

# Button for PDF generation
if st.button("Generate PDF"):
    if st.session_state.processed_frames:
        pdf_file = "pothole_detection_report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Pothole Detection Report")
        c.drawString(100, 730, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        y = 700
        for i, (frame, detections) in enumerate(st.session_state.processed_frames):
            img_path = f"frame_{i}.jpg"
            cv2.imwrite(img_path, frame)
            c.drawString(100, y, f"Frame {i+1}: {len(detections)} potholes detected")
            c.drawImage(img_path, 100, y-150, width=400, height=300)
            y -= 170
            if y < 50:
                c.showPage()
                y = 750
            os.remove(img_path)  # Clean up temporary image
        c.save()
        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_file)
        os.remove(pdf_file)  # Clean up PDF file
    else:
        st.warning("No pothole detections available for PDF.")

# Function to process frames
def process_frame(frame):
    st.session_state.frame_count += 1
    img_resized = cv2.resize(frame, (320, 320))
    success, img_encoded = cv2.imencode('.jpg', img_resized)
    if not success:
        return frame
    
    img_bytes = img_encoded.tobytes()
    
    # Send frame to server
    try:
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"latitude": st.session_state.latitude, "longitude": st.session_state.longitude}
        response = requests.post(f"{api_url}/process_frame/", files=files, data=data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.detections = data.get("detections", [])
        
        # Draw detections
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
        potholes_detected = False
        for det in st.session_state.detections:
            if det["confidence"] > 0.5:  # Confidence threshold
                potholes_detected = True
                x1, y1 = int(det["x_min"] * scale_x), int(det["y_min"] * scale_y)
                x2, y2 = int(det["x_max"] * scale_x), int(det["y_max"] * scale_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Store frame and detections only if potholes are detected
        if potholes_detected:
            st.session_state.processed_frames.append((frame.copy(), st.session_state.detections.copy()))
    except Exception as e:
        st.error(f"Request failed: {e}")
    
    return frame

# WebRTC streamer for live video
def video_frame_callback(frame):
    # Convert WebRTC frame to OpenCV format
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame
    processed_img = process_frame(img)
    
    # Convert back to WebRTC frame
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# WebRTC configuration
webrtc_ctx = webrtc_streamer(
    key="pothole-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display processed frames with potholes
if webrtc_ctx.state.playing:
    st.subheader("Processed Frames with Potholes")
    processed_container = st.empty()  # Placeholder for processed frames
    
    while webrtc_ctx.state.playing:
        if st.session_state.processed_frames:
            recent_frame, recent_detections = st.session_state.processed_frames[-1]
            processed_container.image(
                recent_frame,
                channels="BGR",
                caption=f"Processed Frame {len(st.session_state.processed_frames)}: {len(recent_detections)} potholes detected",
                width=300
            )
        time.sleep(0.1)  # Update display periodically
else:
    st.info("Click 'Start' above to begin webcam streaming.")

# Footer
st.markdown("---")
st.write("Built with Streamlit, OpenCV, ReportLab, and streamlit-webrtc | Deployed on Streamlit Cloud")
