import streamlit as st
import os
# Set OpenCV environment variables before importing cv2
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV import error: {e}")
    st.stop()

import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import logging
import av
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc
from twilio.rest import Client
from dotenv import load_dotenv
import threading
import streamlit.components.v1 as components
import base64
from audio_handling import AudioFrameHandler

#https://github.com/veb-101/Drowsiness-Detection-Using-Mediapipe-Streamlit

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Global variables
detection_running = False
drowsiness_detected = False
ear_value = 0.0
audio_playing = False
browser_audio_enabled = True
audio_handler = AudioFrameHandler(os.path.join("audio", "wake_up.wav"))

# Check if we're running on cloud (no pygame available)
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    st.warning("🌐 Running in cloud mode - using browser-based audio alerts")

def get_audio_base64(audio_file_path):
    """Convert audio file to base64 for browser playback"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return audio_base64
    except Exception as e:
        st.error(f"Error encoding audio file: {e}")
        return None

def play_browser_audio():
    """Play audio using HTML5 Audio API in browser"""
    audio_path = os.path.join("audio", "wake_up.wav")
    
    if os.path.exists(audio_path):
        audio_base64 = get_audio_base64(audio_path)
        if audio_base64:
            # Create HTML5 audio player that auto-plays
            audio_html = f"""
            <audio id="drowsiness-alert" autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            <script>
                var audio = document.getElementById('drowsiness-alert');
                audio.play().then(function() {{
                    console.log('Audio played successfully');
                }}).catch(function(error) {{
                    console.log('Audio play error:', error);
                    // Display error in Streamlit
                    var errorDiv = document.createElement('div');
                    errorDiv.innerHTML = 'Audio Error: ' + error.message;
                    document.body.appendChild(errorDiv);
                }});
            </script>
            """
            components.html(audio_html, height=0)
    else:
        # Fallback to beep sound using Web Audio API
        beep_html = """
        <script>
        function playBeep() {
            if (typeof(AudioContext) !== "undefined" || typeof(webkitAudioContext) !== "undefined") {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                var oscillator = audioCtx.createOscillator();
                var gainNode = audioCtx.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);
                
                oscillator.frequency.value = 800; // 800 Hz frequency
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 1);
                
                oscillator.start(audioCtx.currentTime);
                oscillator.stop(audioCtx.currentTime + 1);
                console.log('Beep played');
            }
        }
        playBeep();
        </script>
        """
        components.html(beep_html, height=0)

def play_wake_up_sound():
    """Play the wake up sound when drowsiness is detected"""
    global audio_playing
    
    if audio_playing:
        return  # Prevent overlapping audio
    
    st.write("DEBUG: Attempting to play wake up sound")
    try:
        audio_playing = True
        
        if PYGAME_AVAILABLE:
            # Use pygame for local development
            audio_path = os.path.join("audio", "wake_up.wav")
            if os.path.exists(audio_path):
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
            else:
                st.warning(f"Audio file not found: {audio_path}")
        else:
            # Use browser-based audio for cloud deployment
            if browser_audio_enabled:
                play_browser_audio()
        
        # Reset audio_playing flag after a short delay
        def reset_audio_flag():
            time.sleep(2)  # Adjust based on audio length
            global audio_playing
            audio_playing = False
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=reset_audio_flag, daemon=True).start()
        
    except Exception as e:
        st.error(f"Error playing audio: {e}")
        audio_playing = False

# Twilio configuration
def setup_twilio():
    """Setup Twilio client for WebRTC"""
    try:
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        
        if not account_sid or not auth_token:
            st.warning("⚠️ Twilio credentials not found. Using public STUN servers.")
            st.info("Make sure your .env file contains: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
            return None
        
        st.success("✅ Twilio credentials found!")
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"⚠️ Twilio setup failed: {e}. Using public STUN servers.")
        return None

def get_eye_aspect_ratio(eye_points):
    """Calculate the eye aspect ratio using MediaPipe landmarks"""
    if len(eye_points) < 6:
        return 0.0
    
    # MediaPipe eye landmarks (6 points for each eye)
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

def drowsiness_detection_mediapipe():
    """Drowsiness detection using MediaPipe (no large .dat file needed)"""
    global detection_running, drowsiness_detected, ear_value
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open webcam")
            return
        
        # Detection parameters
        EAR_THRESHOLD = 0.15
        EAR_CONSEC_FRAMES = 15
        COUNTER = 0
        
        # Create placeholder for video feed
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        detection_running = True
        
        try:
            while detection_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                drowsiness_detected = False
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract eye landmarks (MediaPipe uses different indices)
                        # Left eye landmarks
                        left_eye = []
                        for idx in [362, 385, 387, 263, 373, 380]:  # MediaPipe left eye indices
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            left_eye.append([x, y])
                        
                        # Right eye landmarks
                        right_eye = []
                        for idx in [33, 160, 158, 133, 153, 144]:  # MediaPipe right eye indices
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            right_eye.append([x, y])
                        
                        # Calculate eye aspect ratios
                        left_ear = get_eye_aspect_ratio(left_eye)
                        right_ear = get_eye_aspect_ratio(right_eye)
                        ear = (left_ear + right_ear) / 2.0
                        ear_value = ear
                        
                        # Check for drowsiness
                        if ear < EAR_THRESHOLD:
                            COUNTER += 1
                            if COUNTER >= EAR_CONSEC_FRAMES:
                                drowsiness_detected = True
                                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                # Play wake up sound
                                play_wake_up_sound()
                        else:
                            COUNTER = 0
                        
                        # Draw eye contours
                        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                        
                        # Display EAR value
                        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display video feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                time.sleep(0.1)
        
        except Exception as e:
            st.error(f"Error during detection: {e}")
        finally:
            cap.release()
            detection_running = False

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Video frame callback for WebRTC with drowsiness detection"""
    global drowsiness_detected, ear_value
    
    # Convert frame to numpy array
    image = frame.to_ndarray(format="bgr24")
    
    # Initialize MediaPipe if not already done
    if not hasattr(video_frame_callback, 'face_mesh'):
        video_frame_callback.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        video_frame_callback.ear_threshold = 0.15
        video_frame_callback.ear_consec_frames = 15
        video_frame_callback.counter = 0
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = video_frame_callback.face_mesh.process(image_rgb)
    
    drowsiness_detected = False
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks (MediaPipe uses different indices)
            # Left eye landmarks
            left_eye = []
            for idx in [362, 385, 387, 263, 373, 380]:  # MediaPipe left eye indices
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                left_eye.append([x, y])
            
            # Right eye landmarks
            right_eye = []
            for idx in [33, 160, 158, 133, 153, 144]:  # MediaPipe right eye indices
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                right_eye.append([x, y])
            
            # Calculate eye aspect ratios
            left_ear = get_eye_aspect_ratio(left_eye)
            right_ear = get_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            ear_value = ear
            
            # Check for drowsiness
            if ear < video_frame_callback.ear_threshold:
                video_frame_callback.counter += 1
                if video_frame_callback.counter >= video_frame_callback.ear_consec_frames:
                    drowsiness_detected = True
                    cv2.putText(image, "DROWSINESS DETECTED!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                video_frame_callback.counter = 0
            
            # Draw eye contours
            cv2.polylines(image, [np.array(left_eye)], True, (0, 255, 0), 1)
            cv2.polylines(image, [np.array(right_eye)], True, (0, 255, 0), 1)
            
            # Display EAR value
            cv2.putText(image, f"EAR: {ear:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """Audio frame callback to inject wake-up audio via WebRTC when drowsiness is detected."""
    try:
        return audio_handler.process(frame, play_sound=drowsiness_detected)
    except Exception:
        # In case of any processing error, pass the original frame through
        return frame

def main():
    st.set_page_config(
        page_title="Lightweight Drowsiness Detection",
        page_icon="😴",
        layout="wide"
    )
    
    st.title("😴 Lightweight Drowsiness Detection System")
    st.markdown("**Using MediaPipe + WebRTC with Twilio**")
    st.markdown("---")
    
    # Setup Twilio
    ice_servers = setup_twilio()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Audio settings
    st.sidebar.subheader("🔊 Audio Settings")
    global browser_audio_enabled
    if not PYGAME_AVAILABLE:
        browser_audio_enabled = st.sidebar.checkbox("Enable Browser Audio Alerts", value=True, 
                                                   help="Uses HTML5 Audio API for drowsiness alerts")
        if browser_audio_enabled:
            st.sidebar.info("🌐 Using browser-based audio (HTML5 + Web Audio API)")
        else:
            st.sidebar.warning("🔇 Audio alerts disabled - only visual alerts will be shown")
    else:
        st.sidebar.success("🎵 Using Pygame audio (local mode)")
        browser_audio_enabled = True
    
    # Audio test button
    if st.sidebar.button("🔊 Test Audio Alert"):
        play_wake_up_sound()
        st.sidebar.success("Audio test played!")
    
    # Twilio setup instructions
    if not ice_servers:
        st.sidebar.subheader("🔧 Twilio Setup")
        st.sidebar.markdown("""
        **For better WebRTC performance:**
        
        1. Sign up at [Twilio Console](https://console.twilio.com/)
        2. Get Account SID & Auth Token
        3. Set environment variables:
        ```bash
        export TWILIO_ACCOUNT_SID="your_account_sid"
        export TWILIO_AUTH_TOKEN="your_auth_token"
        ```
        """)
    
    # Detection parameters
    st.sidebar.subheader("Detection Settings")
    ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.3, 0.15, 0.01, 
                                     help="Lower values = more sensitive to drowsiness")
    ear_consec_frames = st.sidebar.slider("Consecutive Frames", 5, 30, 15, 1,
                                         help="Number of consecutive frames before alert")
    
    # Update callback parameters
    if hasattr(video_frame_callback, 'ear_threshold'):
        video_frame_callback.ear_threshold = ear_threshold
        video_frame_callback.ear_consec_frames = ear_consec_frames
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Video Feed")
        
        # WebRTC Streamer with Twilio configuration
        rtc_configuration = {"iceServers": ice_servers} if ice_servers else {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=video_frame_callback,
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("🎥 Webcam is active! Drowsiness detection is running.")
            if ice_servers:
                st.info("✅ Using Twilio TURN servers for better connectivity")
            else:
                st.info("ℹ️ Using public STUN servers")
        else:
            st.info("Click 'START' to begin webcam and drowsiness detection.")
    
    with col2:
        st.header("Detection Status")
        
        # Status display
        if webrtc_ctx.state.playing:
            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            ear_placeholder = st.empty()
            
            # Update status based on global variables
            if drowsiness_detected:
                status_placeholder.error("🚨 DROWSINESS DETECTED! 🚨")
                # Additional visual alert with balloons
                st.balloons()
                # Large warning message
                st.markdown("""
                <div style='background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin: 10px 0;'>
                    ⚠️ WAKE UP! DROWSINESS DETECTED! ⚠️
                </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.success("✅ Awake and Alert")
            
            # Display EAR value
            ear_placeholder.metric("Eye Aspect Ratio (EAR)", f"{ear_value:.3f}")
            
            # EAR interpretation
            if ear_value < ear_threshold:
                st.warning(f"EAR below threshold ({ear_threshold})")
            else:
                st.success(f"EAR above threshold ({ear_threshold})")
        else:
            st.info("Status will appear here when detection is running.")
        
        st.markdown("### Advantages of WebRTC + MediaPipe:")
        features_list = [
            "✅ **Better webcam integration**",
            "✅ **Real-time processing**", 
            "✅ **Lower latency**",
            "✅ **Easy deployment**",
            "✅ **Works in browsers**",
            "✅ **Cross-platform audio alerts**",
            "✅ **Visual + Audio notifications**",
            "✅ **Twilio TURN servers** (if configured)"
        ]
        
        if PYGAME_AVAILABLE:
            features_list.append("🎵 **Pygame audio** (local mode)")
        else:
            features_list.extend([
                "🌐 **HTML5 Audio API** (cloud mode)",
                "🔊 **Web Audio API fallback**",
                "🎨 **Enhanced visual alerts**"
            ])
        
        st.markdown("\n".join(features_list))
        
        st.markdown("### How it works:")
        how_it_works = [
            "- **Eye Aspect Ratio (EAR)**: Measures eye openness",
            "- **Threshold**: Adjustable sensitivity",
            "- **Consecutive Frames**: Configurable alert delay", 
            "- **Real-time**: Continuous monitoring",
            "- **Twilio**: Enhanced WebRTC connectivity"
        ]
        
        if PYGAME_AVAILABLE:
            how_it_works.append("- **Audio Alert**: Plays wake_up.wav using Pygame")
        else:
            how_it_works.extend([
                "- **Browser Audio**: HTML5 Audio API with base64 encoding",
                "- **Fallback Beep**: Web Audio API generated tones", 
                "- **Visual Alerts**: Red flash, balloons, and warning messages"
            ])
        
        st.markdown("\n".join(how_it_works))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ This system is for educational purposes. Always stay alert while driving!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Versions:**  \n"
        f"Streamlit: {st.__version__}  \n"
        f"WebRTC: {st_webrtc_version}  \n"
        f"aiortc: {aiortc.__version__}"
    )

if __name__ == "__main__":
    main() 