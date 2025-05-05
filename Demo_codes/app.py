import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn as nn
from torchvision.models.video import r3d_18
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import tempfile
import datetime
from collections import deque
import random
import io
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

st.set_page_config(layout="wide", page_title="Runway Surface Condition Dashboard")

st.sidebar.title("📋 Menu")

app_mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload & Classify", "Trend Analysis"],
    index=0
)

# Initialize session state for storing uploaded images and classifications
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'classifications' not in st.session_state:
    st.session_state.classifications = []

# Common transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# GRF class mapping
description_map = {
    6: "DRY",
    5: "WET",
    4: "COMPACTED SNOW",
    3: "WET",
    2: "STANDING WATER",
    1: "ICE",
    0: "WET ICE"
}

# 1. Load models
@st.cache_resource
def load_2d_model(path):
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Initialize EfficientNet-B3
        model = models.efficientnet_b3(pretrained=False)
        num_classes = 7
        
        # Determine number of classes from state dict
        if 'model.classifier.1.weight' in state_dict:
            num_classes = state_dict['model.classifier.1.weight'].size(0)
        elif 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].size(0)
        else:
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    num_classes = state_dict[key].size(0)
                    break
        
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            if new_key in model_state_dict and model_state_dict[new_key].shape == v.shape:
                new_state_dict[new_key] = v
            else:
                st.warning(f"Skipping key {k} due to shape mismatch or missing in model.")
        
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
        return model, num_classes
    except Exception as e:
        st.error(f"Failed to load 2D model: {e}")
        raise e

@st.cache_resource
def load_3d_model():
    model = r3d_18(pretrained=True)
    num_classes = 7
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model, num_classes

model_path = r"C:\Users\Gil\Documents\image_classifier\best_model_new.pth"
if not os.path.exists(model_path):
    st.error(f"Model not found at `{model_path}`. Please train and save it first.")
    st.stop()

try:
    model_2d, num_classes_2d = load_2d_model(model_path)
    model_3d, num_classes_3d = load_3d_model()
    st.sidebar.success(f"Model 1 loaded with {num_classes_2d} classes, Model 2 loaded with {num_classes_3d} classes!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Helper functions
def classify_image(img, model):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    return predicted.item(), probabilities.cpu().numpy()

def prepare_video_tensor(image_sequence, num_frames=5):
    if len(image_sequence) < num_frames:
        image_sequence = [image_sequence[0]] * (num_frames - len(image_sequence)) + image_sequence
    else:
        image_sequence = image_sequence[-num_frames:]
    video_tensor = torch.stack(image_sequence).permute(1, 0, 2, 3).unsqueeze(0)
    return video_tensor

def run_trend_analysis_3dcnn(model_3d, image_sequence, num_frames=5):
    model_3d.eval()
    with torch.no_grad():
        video_tensor = prepare_video_tensor(image_sequence, num_frames=num_frames)
        outputs = model_3d(video_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    return predicted

def detect_trend(class_sequence):
    if len(class_sequence) < 2:
        return "Stable"
    if class_sequence[-1] > class_sequence[0]:
        return "Improving"
    elif class_sequence[-1] < class_sequence[0]:
        return "Worsening"
    else:
        return "Stable"

def generate_grf_report(runway, code, trend, forecast, weather, timestamp):
    return {
        "timestamp": timestamp.isoformat() + "Z",
        "location": runway,
        "runway_condition_code": code,
        "runway_condition_description": description_map.get(code, "UNKNOWN"),
        "trend_analysis": trend,
        "forecast_next_5min": forecast,
        "temperature": weather.get('temp'),
        "precipitation": weather.get('precip'),
        "coverage": weather.get('coverage'),
        "icao_snowtam_format": f"RWY {runway} {code} / {description_map.get(code)} / TREND: {trend} / TEMP {weather.get('temp')}C / PRECIP {weather.get('precip')}"
    }

def get_class_names():
    return list(description_map.values())

# MODE 1: UPLOAD & CLASSIFY
if app_mode == "Upload & Classify":
    st.title("🔍 Runway Surface Analysis")
    
    st.subheader("Upload Options")
    upload_option = st.radio("Choose upload method:", ["Individual Images", "Dataset (ZIP file)"])
    
    uploaded_files = []
    if upload_option == "Individual Images":
        uploaded_files = st.file_uploader("Upload runway images", 
                                        type=["jpg", "jpeg", "png"], 
                                        accept_multiple_files=True)
    else:
        dataset_file = st.file_uploader("Upload a ZIP file containing runway images", 
                                      type=["zip"])
        if dataset_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(dataset_file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        if file.lower().endswith(('jpg', 'jpeg', 'png')):
                            file_path = os.path.join(root, file)
                            with open(file_path, "rb") as f:
                                uploaded_files.append(f.read())
    
    if uploaded_files:
        if len(uploaded_files) < 6:
            st.warning("Please upload at least 6 images for better analysis.")
        
        st.session_state.uploaded_images = uploaded_files
        
        classify_button = st.button("Classify Images", type="primary")
        
        if classify_button:
            col1, col2 = st.columns(2)
            results = []
            st.session_state.classifications = []
            
            with st.spinner("Classifying images..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    if isinstance(uploaded_file, bytes):
                        img = Image.open(io.BytesIO(uploaded_file)).convert("RGB")
                        file_name = f"image_{i}.png"
                    else:
                        img = Image.open(uploaded_file).convert("RGB")
                        file_name = uploaded_file.name
                    
                    class_idx, probabilities = classify_image(img, model_2d)
                    class_name = description_map.get(class_idx, f"Unknown ({class_idx})")
                    confidence = probabilities[class_idx] * 100
                    
                    results.append({
                        "Image": file_name,
                        "Prediction": class_name,
                        "Confidence": confidence
                    })
                    st.session_state.classifications.append(class_idx)
                    
                    with col1 if i % 2 == 0 else col2:
                        st.image(img, caption=f"File: {file_name}", width=300)
                        st.write(f"**Prediction:** {class_name}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        prob_df = pd.DataFrame({
                            "Class": list(description_map.values()),
                            "Probability (%)": probabilities * 100
                        }).sort_values("Probability (%)", ascending=False).head(5)
                        st.bar_chart(prob_df.set_index("Class"))
                        st.markdown("---")
            
            st.subheader("Classification Summary")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            st.subheader("Prediction Distribution")
            prediction_counts = results_df["Prediction"].value_counts()
            st.bar_chart(prediction_counts)

# MODE 2: TREND ANALYSIS
else:
    st.title("📈 Runway Surface Trend Analysis")
    
    if not st.session_state.uploaded_images:
        st.warning("No images available for trend analysis. Please upload images in the 'Upload & Classify' mode first.")
    else:
        if len(st.session_state.uploaded_images) < 5:
            st.warning("Please upload at least 5 images in 'Upload & Classify' for trend analysis.")
        
        weather_data = {
            'temp': st.number_input("Temperature (°C)", value=2.5),
            'precip': st.selectbox("Precipitation", ["None", "Light snow", "Heavy snow", "Rain", "Sleet"]),
            'coverage': st.slider("Coverage (%)", 0, 100, 65)
        }
        
        runway_id = st.text_input("Runway ID", "01A")
        
        analyze_button = st.button("Analyze Trends", type="primary")
        
        if analyze_button:
            with st.spinner("Analyzing runway condition trends..."):
                image_sequence = []
                for uploaded_file in st.session_state.uploaded_images:
                    if isinstance(uploaded_file, bytes):
                        img = Image.open(io.BytesIO(uploaded_file)).convert("RGB")
                    else:
                        img = Image.open(uploaded_file).convert("RGB")
                    image_sequence.append(transform(img))
                
                grfs = []
                with torch.no_grad():
                    for img in image_sequence:
                        output = model_2d(img.unsqueeze(0))
                        grf_code = torch.argmax(output, dim=1).item()
                        grfs.append(grf_code)
                
                trend_code = run_trend_analysis_3dcnn(model_3d, image_sequence)
                
                results = []
                trend_classes = [run_trend_analysis_3dcnn(model_3d, image_sequence[max(0,i-4):i+1], num_frames=5) 
                               for i in range(len(image_sequence))]
                
                # Generate unique timestamps for each image
                base_time = datetime.datetime.utcnow()
                timestamps = [base_time + datetime.timedelta(minutes=i) for i in range(len(st.session_state.uploaded_images))]
                
                for i, (code, timestamp) in enumerate(zip(grfs, timestamps)):
                    trend_window = trend_classes[max(0, i-4):i+1]
                    trend_label = detect_trend(trend_window)
                    report = generate_grf_report(runway_id, code, trend_label, trend_code, weather_data, timestamp)
                    results.append(report)
                
                st.subheader("Runway Condition Reports")
                for i, (report, uploaded_file) in enumerate(zip(results, st.session_state.uploaded_images)):
                    if isinstance(uploaded_file, bytes):
                        img = Image.open(io.BytesIO(uploaded_file)).convert("RGB")
                    else:
                        img = Image.open(uploaded_file).convert("RGB")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(img, width=200)
                    with col2:
                        st.write(f"**Timestamp:** {report['timestamp']}")
                        st.write(f"**Runway:** {report['location']}")
                        st.write(f"**GRF Code:** {report['runway_condition_code']} ({report['runway_condition_description']})")
                        st.write(f"**Trend:** {report['trend_analysis']}")
                        st.write(f"**Forecast (5min):** {report['forecast_next_5min']}")
                        st.write(f"**Weather:** Temp {report['temperature']}°C, {report['precipitation']}, {report['coverage']}% coverage")
                        st.write(f"**ICAO SNOWTAM:** {report['icao_snowtam_format']}")
                        st.markdown("---")
                
                # Plot trends
                st.subheader("Condition Trend Analysis")
                fig, ax = plt.subplots(figsize=(12, 6))
                codes = [r['runway_condition_code'] for r in results]
                timestamp_strings = [r['timestamp'] for r in results]
                
                ax.plot(timestamp_strings, codes, label='Observed GRF Code', marker='o')
                ax.plot(timestamp_strings, [trend_code] * len(timestamp_strings), 
                       label='Forecast Code', linestyle='-.', marker='s', color='green')
                
                # Add class labels to data points
                for i, code in enumerate(codes):
                    ax.annotate(description_map.get(code, "UNKNOWN"),
                               (timestamp_strings[i], code),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center')
                
                ax.set_xticklabels(timestamp_strings, rotation=45, ha='right')
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("GRF Code")
                ax.set_title("Runway Condition Trends")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)


## Usage
"""
Run this in the terminal: 

streamlit run app.py

"""

