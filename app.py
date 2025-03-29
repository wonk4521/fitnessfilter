
import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# === Streamlit Fitness Filter Web App ===

# --- Pose Detection Function ---
def detect_pose(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(image)

    return results.pose_landmarks

# --- Apply Visual Slimming Filter ---
def apply_fitness_filter(image, intensity=0.15):
    height, width = image.shape[:2]
    new_width = int(width * (1 - intensity))
    slimmed = cv2.resize(image, (new_width, height))
    result = cv2.resize(slimmed, (width, height))

    return result

# --- Streamlit App ---
st.title("🏋️‍♂️ Fitness Transformation Preview")
st.write("Upload a photo and adjust the workout frequency slider to see a simulated fitness transformation.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

intensity = st.slider("Workout Frequency (Higher = More Fit Look)", 0.0, 0.30, 0.15, 0.01)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pose detection (optional visualization)
    landmarks = detect_pose(image_rgb)
    if landmarks:
        st.write("Pose detected.")
    else:
        st.write("No pose detected.")

    # Apply transformation
    transformed = apply_fitness_filter(image, intensity=intensity)
    transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

    st.image([image_rgb, transformed_rgb], caption=["Original", "Fitness Preview"], width=300)

    # Download button
    result_filename = "fitness_preview.jpg"
    cv2.imwrite(result_filename, transformed)
    with open(result_filename, "rb") as file:
        st.download_button(
            label="Download Fitness Preview",
            data=file,
            file_name=result_filename,
            mime="image/jpeg"
        )
