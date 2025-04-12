import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import requests
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from typing import List, Tuple

# Initialize session state
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames: List[Tuple[np.ndarray, list]] = []  # (frame, detections)
if "api_url" not in st.session_state:
    st.session_state.api_url = "https://1fd0-2402-3a80-4273-e6c3-f0df-5dc6-4ca5-5b58.ngrok-free.app"

# WebRTC configuration (STUN server for NAT traversal)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class PotholeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detections = []
        self.frame_count = 0
        self.should_capture = False
    
    def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process frame only if capture is enabled
        if self.should_capture:
            processed_img, detections = self._process_image(img)
            if detections:  # Only store frames with detections
                st.session_state.processed_frames.append((processed_img.copy(), detections.copy()))
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        return frame
    
    def _process_image(self, img: np.ndarray) -> Tuple[np.ndarray, list]:
        """Send image to API and process detections"""
        img_resized = cv2.resize(img, (320, 320))
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        img_bytes = img_encoded.tobytes()
        
        detections = []
        try:
            files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
            response = requests.post(
                f"{st.session_state.api_url}/process_frame/",
                files=files,
                data={"latitude": None, "longitude": None},
                timeout=5
            )
            if response.status_code == 200:
                detections = response.json().get("detections", [])
        except Exception as e:
            print(f"API Error: {e}")
        
        # Draw detections
        scale_x = img.shape[1] / 320
        scale_y = img.shape[0] / 320
        for det in detections:
            if det["confidence"] > 0.5:
                x1 = int(det["x_min"] * scale_x)
                y1 = int(det["y_min"] * scale_y)
                x2 = int(det["x_max"] * scale_x)
                y2 = int(det["y_max"] * scale_y)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return img, detections

def generate_pdf():
    """Generate PDF from processed frames"""
    if not st.session_state.processed_frames:
        st.warning("No pothole detections available for PDF.")
        return None
    
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
        if y < 50:  # New page if we run out of space
            c.showPage()
            y = 750
        os.remove(img_path)
    
    c.save()
    return pdf_file

# Streamlit UI
st.title("Real-time Pothole Detection with WebRTC")

# API URL configuration
st.session_state.api_url = st.text_input(
    "Server API URL",
    st.session_state.api_url
)

# WebRTC Streamer
ctx = webrtc_streamer(
    key="pothole-detection",
    video_processor_factory=PotholeVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

# Capture control
if ctx.video_processor:
    if st.button("Start/Stop Capture"):
        ctx.video_processor.should_capture = not ctx.video_processor.should_capture
        st.session_state.capturing = ctx.video_processor.should_capture

# Display processed frames
st.subheader("Detected Potholes")
if st.session_state.processed_frames:
    cols = st.columns(3)
    for idx, (frame, detections) in enumerate(st.session_state.processed_frames[-6:]):  # Show last 6
        with cols[idx % 3]:
            st.image(frame, channels="BGR", caption=f"{len(detections)} potholes")
else:
    st.info("No potholes detected yet")

# PDF Generation
if st.button("Generate PDF Report"):
    pdf_file = generate_pdf()
    if pdf_file:
        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name=pdf_file,
                mime="application/pdf"
            )
        os.remove(pdf_file)
