import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import requests
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from typing import List, Tuple, Optional

# Configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
DEFAULT_API_URL = "http://localhost:8000"  # Update with your deployed URL

# Session State
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames: List[dict] = []
if "location" not in st.session_state:
    st.session_state.location = {"latitude": 0.0, "longitude": 0.0}

class PotholeDetector(VideoProcessorBase):
    def __init__(self):
        self.capture_active = False
        self.frame_count = 0
    
    def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.capture_active:
            processed_img, result = process_image_with_api(
                img, 
                st.session_state.api_url,
                st.session_state.location
            )
            
            if result and result.get("detections"):
                st.session_state.processed_frames.append({
                    "image": processed_img,
                    "detections": result["detections"],
                    "location": result["location"],
                    "timestamp": result["timestamp"]
                })
                
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        
        return frame

def process_image_with_api(img: np.ndarray, api_url: str, location: dict) -> Tuple[np.ndarray, Optional[dict]]:
    """Send image to API for processing"""
    try:
        # Resize and encode image
        img_resized = cv2.resize(img, (640, 640))
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        img_bytes = img_encoded.tobytes()
        
        # Prepare request
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(
            f"{api_url}/process_frame/",
            files=files,
            data=location,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", [])
            
            # Draw detections
            scale_x = img.shape[1] / 640
            scale_y = img.shape[0] / 640
            for det in detections:
                x1 = int(det["x_min"] * scale_x)
                y1 = int(det["y_min"] * scale_y)
                x2 = int(det["x_max"] * scale_x)
                y2 = int(det["y_max"] * scale_y)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            return img, result
            
    except Exception as e:
        st.error(f"API Error: {e}")
    
    return img, None

def generate_pdf_report():
    """Generate PDF report from detected potholes"""
    if not st.session_state.processed_frames:
        return None
    
    pdf_path = "pothole_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # PDF Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Pothole Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y_position = 700
    for idx, detection in enumerate(st.session_state.processed_frames):
        # Save frame as temp image
        img_path = f"temp_{idx}.jpg"
        cv2.imwrite(img_path, detection["image"])
        
        # Add to PDF
        c.drawString(100, y_position, f"Detection #{idx+1}")
        c.drawString(100, y_position-20, f"Location: {detection['location']}")
        c.drawString(100, y_position-40, f"Time: {detection['timestamp']}")
        c.drawString(100, y_position-60, f"Potholes detected: {len(detection['detections']}")
        
        # Add image
        c.drawImage(img_path, 100, y_position-300, width=400, height=300)
        
        y_position -= 350
        if y_position < 100:
            c.showPage()
            y_position = 750
        
        os.remove(img_path)
    
    c.save()
    return pdf_path

def main():
    st.title("🚧 Real-Time Pothole Detection System")
    
    # Settings Sidebar
    with st.sidebar:
        st.header("Settings")
        st.session_state.api_url = st.text_input(
            "API Server URL", 
            st.session_state.api_url
        )
        
        st.header("Location")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.location["latitude"] = st.number_input(
                "Latitude", 
                value=st.session_state.location["latitude"],
                format="%.6f"
            )
        with col2:
            st.session_state.location["longitude"] = st.number_input(
                "Longitude", 
                value=st.session_state.location["longitude"],
                format="%.6f"
            )
        
        if st.button("Use Current Location (Browser)"):
            st.warning("Enable location access in your browser settings")
    
    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="pothole-detection",
        video_processor_factory=PotholeDetector,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
        async_processing=True,
    )
    
    # Capture Control
    if ctx.video_processor:
        if st.button("Toggle Capture Mode"):
            ctx.video_processor.capture_active = not ctx.video_processor.capture_active
            status = "ON" if ctx.video_processor.capture_active else "OFF"
            st.success(f"Capture mode {status}")
    
    # Detection Display
    st.header("Detected Potholes")
    if st.session_state.processed_frames:
        cols = st.columns(2)
        for idx, detection in enumerate(st.session_state.processed_frames[-4:]):
            with cols[idx % 2]:
                st.image(
                    detection["image"], 
                    channels="BGR",
                    caption=f"{len(detection['detections'])} potholes at {detection['location']}"
                )
    else:
        st.info("No potholes detected yet. Enable capture mode to start detection.")
    
    # Report Generation
    st.header("Report Generation")
    if st.button("Generate PDF Report"):
        report_path = generate_pdf_report()
        if report_path:
            with open(report_path, "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name="pothole_detection_report.pdf",
                    mime="application/pdf"
                )
            os.remove(report_path)
        else:
            st.warning("No detections available for report")

if __name__ == "__main__":
    main()
