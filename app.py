import streamlit as st
import cv2
import numpy as np
import requests
import time
from reportlab.lib.pagesizes import import letter
from reportlab.pdfgen import canvas
import os
import tempfile

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
st.write("Upload an image or video to detect potholes. Supported formats: JPG, PNG, MP4.")

# Input for server API URL
api_url = st.text_input("Server API URL", "https://1fd0-2402-3a80-4273-e6c3-f0df-5dc6-4ca5-5b58.ngrok-free.app")

# File uploader for images or videos
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "png", "mp4"])

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

# Process uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.type
    st.subheader("Uploaded Content")
    content_container = st.empty()  # Placeholder for uploaded content
    st.subheader("Processed Frames with Potholes")
    processed_container = st.empty()  # Placeholder for processed frames
    
    try:
        # Handle image
        if file_type in ["image/jpeg", "image/png"]:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                st.error("Failed to load the image.")
            else:
                # Display original image
                content_container.image(frame, channels="BGR", caption="Uploaded Image")
                
                # Process the frame
                processed_frame = process_frame(frame)
                
                # Update content container with processed frame
                content_container.image(
                    processed_frame,
                    channels="BGR",
                    caption=f"Processed Image: {len(st.session_state.detections)} potholes detected"
                )
                
                # Update processed frames display if potholes were detected
                if st.session_state.processed_frames:
                    recent_frame, recent_detections = st.session_state.processed_frames[-1]
                    processed_container.image(
                        recent_frame,
                        channels="BGR",
                        caption=f"Processed Frame {len(st.session_state.processed_frames)}: {len(recent_detections)} potholes detected",
                        width=300
                    )
        
        # Handle video
        elif file_type == "video/mp4":
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(tmp_file_path)
            if not cap.isOpened():
                st.error("Failed to open the video.")
            else:
                st.write("Processing video... This may take a moment.")
                frame_placeholder = st.empty()  # Placeholder for video frames
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process the frame
                    processed_frame = process_frame(frame)
                    
                    # Display current frame
                    frame_placeholder.image(
                        processed_frame,
                        channels="BGR",
                        caption=f"Frame {st.session_state.frame_count}: {len(st.session_state.detections)} potholes detected"
                    )
                    
                    # Update processed frames display if potholes were detected
                    if st.session_state.processed_frames:
                        recent_frame, recent_detections = st.session_state.processed_frames[-1]
                        processed_container.image(
                            recent_frame,
                            channels="BGR",
                            caption=f"Processed Frame {len(st.session_state.processed_frames)}: {len(recent_detections)} potholes detected",
                            width=300
                        )
                    
                    time.sleep(0.1)  # Simulate real-time processing
                
                cap.release()
                st.success("Video processing complete.")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        else:
            st.error("Unsupported file type. Please upload a JPG, PNG, or MP4 file.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload an image or video to start pothole detection.")

# Footer
st.markdown("---")
st.write("Built with Streamlit, OpenCV, and ReportLab | Deployed on Streamlit Cloud")
