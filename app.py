import streamlit as st
import subprocess
import sys
import tempfile
from face_auth import register_face, start_face_auth_camera

st.set_page_config(page_title="AI Vision Dashboard", layout="wide")
st.title("🧠 AI Vision Dashboard")

st.sidebar.title("Navigation")
feature = st.sidebar.radio(
    "Select Feature",
    ["AI Bag Counter", "Face Authentication", "Wall Area Estimation"]
)

# =====================================
#   AI BAG COUNTER UI
# =====================================

if feature == "AI Bag Counter":

    st.header("👜 AI Bag Counter")

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT - Live Camera
    # -------------------------
    with col1:
        st.subheader("🔵 Live Camera")

        if st.button("Start Bag Counter Camera"):
            subprocess.run([sys.executable, "ai_bag_counter.py"])

    # -------------------------
    # RIGHT - Upload Video
    # -------------------------
    with col2:
        st.subheader("🟢 Upload Video for Testing")

        uploaded_video = st.file_uploader(
            "Upload Conveyor Belt Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video is not None:

            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_video.read())

            if st.button("Process Uploaded Video"):
                subprocess.run([
                    sys.executable,
                    "ai_bag_counter.py",
                    temp_file.name
                ])

# =====================================
# FACE AUTHENTICATION UI
# =====================================
if feature == "Face Authentication":

    st.header("😀 Face Authentication")

    col1, col2 = st.columns(2)

    # -------------------------
    # Left SIDE - AUTH CAMERA
    # -------------------------
    with col1:
        st.subheader("🔵 Start Authentication")

        if st.button("Start Live Camera"):
            start_face_auth_camera()
    
    # -------------------------
    # Right SIDE - REGISTER
    # -------------------------
    with col2:
        st.subheader("🟢 Register New Face")

        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            if st.button("Register Face"):
                message = register_face(temp_file.name)
                st.success(message)

# =====================================
# WALL AREA ESTIMATION UI
# =====================================
if feature == "Wall Area Estimation":

    st.header("🎨 Wall Area Estimation")

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT - Live Camera
    # -------------------------
    with col1:
        st.subheader("🔵 Live Camera")

        if st.button("Start Wall Area Camera"):
            subprocess.run([sys.executable, "wall_area.py"])

    # -------------------------
    # RIGHT - Upload Image
    # -------------------------
    with col2:
        st.subheader("🟢 Upload Image for Testing")

        uploaded_image = st.file_uploader(
            "Upload Wall Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )
            temp_file.write(uploaded_image.read())

            st.image(uploaded_image, caption="Uploaded Image", width="stretch")

            if st.button("Process Uploaded Image"):
                subprocess.run([
                    sys.executable,
                    "wall_area.py",
                    temp_file.name
                ])