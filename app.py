import streamlit as st
import cv2
import numpy as np
import requests
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables
if "streaming" not in st.session_state:
    st.session_state.streaming = False
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
if "webrtc_error" not in st.session_state:
    st.session_state.webrtc_error = False
if "api_status" not in st.session_state:
    st.session_state.api_status = None  # None, "connected", or "failed"
if "frames_sent" not in st.session_state:
    st.session_state.frames_sent = 0
if "frames_responded" not in st.session_state:
    st.session_state.frames_responded = 0
if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = 0
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None

# Streamlit page configuration
st.set_page_config(page_title="Pothole Detection App", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .video-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .video-box h3 {
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .processed-box {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 10px;
        background-color: #f0f8ff;
    }
    .processed-box h3 {
        color: #2196F3;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Pothole Detection App")
st.write("Stream live video from your webcam to detect potholes in real-time.")

# Input for server API URL
api_url = st.text_input("Server API URL", "https://1077-2402-3a80-54b-55a6-ac0f-c66d-7b7e-fca4.ngrok-free.app")

# API connection check
def check_api_connection():
    try:
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        success, img_encoded = cv2.imencode('.jpg', dummy_img)
        if not success:
            st.session_state.api_status = "failed"
            st.error("Failed to encode test image.")
            return
        
        img_bytes = img_encoded.tobytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{api_url}/process_frame/", files=files, timeout=5)
        
        if response.status_code == 200:
            st.session_state.api_status = "connected"
            st.success("API is connected and responsive!")
        else:
            st.session_state.api_status = "failed"
            st.error(f"API returned status {response.status_code}. Please check the URL or server.")
    except requests.exceptions.RequestException as e:
        st.session_state.api_status = "failed"
        st.error(f"Failed to connect to API: {str(e)}")

if st.button("Check API Connection"):
    check_api_connection()

# Display API status
if st.session_state.api_status == "connected":
    st.success("API Status: Connected")
elif st.session_state.api_status == "failed":
    st.error("API Status: Not Connected")

# Placeholder for frame status
frame_status = st.empty()

# Display frame transmission stats
st.metric(label="Frames Sent to API", value=st.session_state.frames_sent)
st.metric(label="Frames Responded by API", value=st.session_state.frames_responded)

# Buttons for streaming and PDF generation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start Video Stream"):
        st.session_state.streaming = True
        st.session_state.webrtc_error = False
with col2:
    if st.button("Stop Video Stream"):
        st.session_state.streaming = False
with col3:
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
                os.remove(img_path)
            c.save()
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_file)
            os.remove(pdf_file)
        else:
            st.warning("No pothole detections available for PDF. Try capturing frames with potholes.")

# Function to process frames (identical to OpenCV version)
def process_frame(frame):
    st.session_state.frame_count += 1
    logger.info(f"Processing frame {st.session_state.frame_count}, shape={frame.shape}")
    
    # Check frame validity
    if frame is None or frame.size == 0:
        logger.error("Invalid frame received")
        frame_status.warning(f"Frame {st.session_state.frame_count}: Invalid frame")
        return frame
    
    img_resized = cv2.resize(frame, (320, 320))
    success, img_encoded = cv2.imencode('.jpg', img_resized)
    if not success:
        logger.error("Failed to encode frame to JPG")
        frame_status.error(f"Frame {st.session_state.frame_count}: Failed to encode")
        return frame
    
    img_bytes = img_encoded.tobytes()
    
    # Rate limit: send every 0.5 seconds
    current_time = time.time()
    if current_time - st.session_state.last_frame_time < 0.5:
        return frame
    
    # Send frame to server
    try:
        st.session_state.frames_sent += 1
        logger.info(f"Sending frame {st.session_state.frames_sent} to API")
        frame_status.info(f"Frame {st.session_state.frames_sent}: Sending to API")
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"latitude": st.session_state.latitude, "longitude": st.session_state.longitude}
        response = requests.post(f"{api_url}/process_frame/", files=files, data=data, timeout=10)
        st.session_state.last_frame_time = current_time
        
        if response.status_code == 200:
            st.session_state.frames_responded += 1
            logger.info(f"Received response for frame {st.session_state.frames_responded}")
            data = response.json()
            st.session_state.detections = data.get("detections", [])
            if st.session_state.detections:
                frame_status.success(f"Frame {st.session_state.frames_responded}: {len(st.session_state.detections)} potholes detected")
            else:
                frame_status.info(f"Frame {st.session_state.frames_responded}: Processed, no potholes detected")
        else:
            logger.warning(f"API returned status {response.status_code} for frame {st.session_state.frames_sent}")
            frame_status.error(f"Frame {st.session_state.frames_sent}: API returned status {response.status_code}")
        
        # Draw detections
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
        potholes_detected = False
        for det in st.session_state.detections:
            if det["confidence"] > 0.3:  # Lowered threshold for testing
                potholes_detected = True
                x1, y1 = int(det["x_min"] * scale_x), int(det["y_min"] * scale_y)
                x2, y2 = int(det["x_max"] * scale_x), int(det["y_max"] * scale_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if potholes_detected:
            st.session_state.processed_frames.append((frame.copy(), st.session_state.detections.copy()))
    except requests.exceptions.Timeout:
        logger.error(f"Timeout for frame {st.session_state.frames_sent}")
        frame_status.error(f"Frame {st.session_state.frames_sent}: API timeout")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for frame {st.session_state.frames_sent}")
        frame_status.error(f"Frame {st.session_state.frames_sent}: API connection failed")
    except Exception as e:
        logger.error(f"Request failed for frame {st.session_state.frames_sent}: {str(e)}")
        frame_status.error(f"Frame {st.session_state.frames_sent}: Failed ({str(e)})")
    
    return frame

# WebRTC streamer for live video
def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        st.session_state.latest_frame = processed_img  # Store for display
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    except Exception as e:
        st.session_state.webrtc_error = True
        logger.error(f"WebRTC error: {str(e)}")
        st.error(f"WebRTC processing error: {str(e)}")
        return frame

# WebRTC configuration
rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    }
)

# Video container and processed frames display
if st.session_state.streaming:
    with st.container():
        st.markdown('<div class="video-box"><h3>Live Video Stream</h3></div>', unsafe_allow_html=True)
        video_container = st.empty()  # Placeholder for live video
    
    with st.container():
        st.markdown('<div class="processed-box"><h3>Processed Frames with Potholes</h3></div>', unsafe_allow_html=True)
        processed_container = st.empty()  # Placeholder for processed frames
    
    try:
        webrtc_ctx = webrtc_streamer(
            key=f"pothole-detection-{int(time.time())}",  # Unique key to reset stream
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Update display while streaming
        while webrtc_ctx and webrtc_ctx.state.playing and st.session_state.streaming and not st.session_state.webrtc_error:
            if st.session_state.latest_frame is not None:
                video_container.image(
                    st.session_state.latest_frame,
                    channels="BGR",
                    caption=f"Frame {st.session_state.frame_count}",
                    use_column_width=True
                )
            
            if st.session_state.processed_frames:
                recent_frame, recent_detections = st.session_state.processed_frames[-1]
                processed_container.image(
                    recent_frame,
                    channels="BGR",
                    caption=f"Processed Frame {len(st.session_state.processed_frames)}: {len(recent_detections)} potholes detected",
                    width=300
                )
            
            time.sleep(0.1)  # Control display frame rate
    
    except Exception as e:
        st.session_state.webrtc_error = True
        logger.error(f"WebRTC init error: {str(e)}")
        st.error(f"Failed to initialize WebRTC stream: {str(e)}")

elif st.session_state.webrtc_error:
    st.warning("WebRTC stream failed. Please refresh the page or check your network and camera settings.")
else:
    st.info("Click 'Start Video Stream' to begin. Ensure your browser has camera access.")

# Footer
st.markdown("---")
st.write("Built with Streamlit, OpenCV, ReportLab, and streamlit-webrtc | Deployed on Streamlit Cloud")
