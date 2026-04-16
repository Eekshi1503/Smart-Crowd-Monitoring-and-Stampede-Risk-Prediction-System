import streamlit as st
import cv2
import numpy as np
import datetime
import time
import base64
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crowd Monitoring",
    layout="wide"
)

# ---------------- LOAD BACKGROUND ----------------
def load_bg(img):
    with open(img, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = load_bg("image.jpg")   # keep your image in same folder

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>

[data-testid="stSidebar"]{{display:none;}}
header, footer {{visibility:hidden;}}

.stApp {{
background-image: url("data:image/jpg;base64,{bg}");
background-size: cover;
background-position: center;
}}

h1,h2,h3,p,label,span {{
color:white !important;
}}

input {{
color:black !important;
}}

div[data-testid="stFileUploader"] span {{
color:black !important;
}}

div[data-testid="stFileUploader"] small {{
color:black !important;
}}

button {{
background:#00c8ff !important;
color:white !important;
border-radius:10px !important;
font-size:16px !important;
padding:10px 25px !important;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD YOLO ----------------
@st.cache_resource
def load_model():
    return YOLO("crowd_head_yolov8n.pt")

model = load_model()

# ---------------- FUNCTIONS ----------------
def head_count(frame):
    results = model(frame, conf=0.3, verbose=False)
    if results and results[0].boxes is not None:
        classes = results[0].boxes.cls.cpu().numpy()
        return int((classes == 0).sum())
    return 0

def density(count, capacity):
    return min((count / capacity) * 100, 100)

def risk_level(d):
    if d < 30:
        return "LOW"
    elif d < 70:
        return "MEDIUM"
    return "HIGH"

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center'>Smart Crowd Monitoring & Stampede Risk Prediction System</h1>", unsafe_allow_html=True)

# ---------------- INPUT MODE ----------------
mode = st.radio("Select Input Mode", ["Image", "Webcam"])

capacity = st.number_input("Maximum Area Capacity", min_value=1, value=1000)

now = datetime.datetime.now()
st.write(f"Date: {now.strftime('%d-%m-%Y')}")
st.write(f"Time: {now.strftime('%H:%M:%S')}")

# ---------------- IMAGE MODE ----------------
if mode == "Image":

    file = st.file_uploader("Upload Image", ["jpg","jpeg","png"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        count = head_count(img)
        dens = density(count, capacity)
        risk = risk_level(dens)

        st.subheader("Results")
        st.write(f"People Count: {count}")
        st.write(f"Density: {dens:.2f}%")
        if risk == "LOW":
            st.markdown("<h4 style='color:#00ff00'>Risk Level: LOW</h4>", unsafe_allow_html=True)
        elif risk == "MEDIUM":
            st.markdown("<h4 style='color:#ffcc00'>Risk Level: MEDIUM</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color:#ff3333'>Risk Level: HIGH</h4>", unsafe_allow_html=True)
        if count > capacity:
            st.markdown(
                "<h4 style='color:red'>⚠ ALERT: Crowd exceeds maximum capacity!</h4>",
                unsafe_allow_html=True
            )

# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":

    st.info("Capturing every 3 seconds for 15 seconds")

    cam = cv2.VideoCapture(0)
    frame_box = st.empty()
    counts = []

    start = time.time()
    last = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if time.time() - last >= 3:
            counts.append(head_count(frame))
            last = time.time()

        if time.time() - start > 15:
            break

    cam.release()

    if counts:
        avg = int(np.mean(counts))
        dens = density(avg, capacity)
        risk = risk_level(dens)

        st.subheader("Results")
        st.write(f"People Count: {avg}")
        st.write(f"Density: {dens:.2f}%")
        st.write(f"Risk Level: {risk}")
