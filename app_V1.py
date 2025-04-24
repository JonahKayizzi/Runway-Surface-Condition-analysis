import streamlit as st
import torch
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import tempfile
import shutil

st.set_page_config(layout="wide", page_title="Image Classification Dashboard")

st.sidebar.title("📋 Menu")

app_mode = st.sidebar.radio(
    "Choose Mode",
    ["Model Report", "Upload & Classify"],
    index=0
)


# 1. Load the model (common for both modes)
@st.cache_resource
def load_model(path):
    # Load the state dict first to check the number of classes
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    # Get the number of classes from the classifier's bias dimensions
    num_classes = state_dict['classifier.1.bias'].size(0)
    
    # Initialize the model with the correct number of classes
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, num_classes

# Define the image transformation (common for both modes)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_path = r"C:\Users\Gil\Documents\image_classifier\latest_model.pt"
if not os.path.exists(model_path):
    st.error(f"Model not found at `{model_path}`. Please train and save it first.")
    st.stop()

try:
    model, num_classes = load_model(model_path)
    st.sidebar.success(f"Model loaded with {num_classes} output classes!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Helper function to classify a single image
def classify_image(img):
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        
    return predicted.item(), probabilities.cpu().numpy()

# Function to extract class names from model
@st.cache_data
def get_class_names(model_dir=r"C:\Users\Gil\Documents\image_classifier"):
    # Try to find a classes.txt file
    classes_file = os.path.join(model_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    # If not found, try to find a test directory with class folders
    test_dir = os.path.join(model_dir, "test")
    if os.path.exists(test_dir):
        return sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    # If nothing found, generate generic class names
    return [f"Class {i}" for i in range(num_classes)]

# Get class names
class_names = get_class_names()

# MODE 1: MODEL REPORT
if app_mode == "Model Report":
    st.title("📊 Image Classification Report Dashboard")
    
    # 2. Prepare test dataset
    test_dir = r"C:\Users\Gil\Documents\image_classifier\test"
    if not os.path.exists(test_dir):
        st.error(f"Test directory not found at `{test_dir}`.")
        st.stop()

    try:
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        st.success(f"Test dataset loaded with {len(test_dataset)} images and {len(test_dataset.classes)} classes.")
        
        # Display warning if number of classes doesn't match
        if len(test_dataset.classes) != num_classes:
            st.warning(f"⚠️ Warning: The model has {num_classes} output classes, but the test dataset has {len(test_dataset.classes)} classes. This may cause issues.")
    except Exception as e:
        st.error(f"Error loading test dataset: {e}")
        st.stop()

    # 3. Run predictions and collect true/predicted labels
    true_labels = []
    pred_labels = []

    with st.spinner("Running inference on test dataset..."):
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
        st.success("Inference completed!")

    # 4. Generate and display classification report
    target_names = test_dataset.classes
    report_dict = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df.style.format(precision=2))

    # Add a section to display class distribution
    st.subheader("Class Distribution in Test Set")
    class_counts = pd.Series([true_labels.count(i) for i in range(len(target_names))], index=target_names)
    st.bar_chart(class_counts)

    # Add a confusion matrix visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names, 
           yticklabels=target_names,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    st.pyplot(fig)

    # Add per-class metrics visualization
    st.subheader("Per-Class Performance")
    per_class = report_df.iloc[:-3]
    metrics = ['precision', 'recall', 'f1-score']

    # Create a multi-bar chart for class metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(per_class))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width - width, per_class[metric], width, label=metric)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(per_class.index, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

# MODE 2: UPLOAD & CLASSIFY
elif app_mode == "Upload & Classify":
    st.title("🔍 Image Classification")
    
    upload_option = st.radio("Choose upload method:", 
                           ["Upload Individual Images", "Upload Dataset (ZIP)"],
                           horizontal=True)
    
    if upload_option == "Upload Individual Images":
        # Allow multiple image upload
        uploaded_files = st.file_uploader("Upload images to classify", 
                                         type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            if len(uploaded_files) < 6:
                st.warning("Please upload at least 6 images for better analysis.")
            
            # Add a button to classify
            classify_button = st.button("Classify Images", type="primary")
            
            if classify_button:
                col1, col2 = st.columns(2)
                
                # Setup for results tracking
                results = []
                
                with st.spinner("Classifying images..."):
                    # Process each image
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Open the image
                        img = Image.open(uploaded_file)
                        
                        # Classify the image
                        class_idx, probabilities = classify_image(img)
                        
                        # Get the class name and probability
                        if class_idx < len(class_names):
                            class_name = class_names[class_idx]
                        else:
                            class_name = f"Unknown ({class_idx})"
                        
                        confidence = probabilities[class_idx] * 100
                        
                        # Store results
                        results.append({
                            "Image": uploaded_file.name,
                            "Prediction": class_name,
                            "Confidence": confidence
                        })
                        
                        # Display in columns
                        with col1 if i % 2 == 0 else col2:
                            st.image(img, caption=f"File: {uploaded_file.name}", width=300)
                            st.write(f"**Prediction:** {class_name}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            
                            # Create a bar chart for top 5 probabilities
                            top_indices = np.argsort(probabilities)[-5:][::-1]
                            top_probs = probabilities[top_indices] * 100
                            top_classes = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in top_indices]
                            
                            prob_df = pd.DataFrame({
                                "Class": top_classes,
                                "Probability (%)": top_probs
                            })
                            
                            st.bar_chart(prob_df.set_index("Class"))
                            st.markdown("---")
                
                # Display summary table
                st.subheader("Classification Summary")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Display class distribution of predictions
                st.subheader("Prediction Distribution")
                prediction_counts = results_df["Prediction"].value_counts()
                st.bar_chart(prediction_counts)
    
    else:  # Upload Dataset (ZIP)
        uploaded_zip = st.file_uploader("Upload a ZIP file containing image folders", type=["zip"])
        
        if uploaded_zip:
            # Create a temporary directory to extract the zip
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract the zip file
                with zipfile.ZipFile(uploaded_zip) as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                # Count image files
                image_count = 0
                image_exts = ['.jpg', '.jpeg', '.png']
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_exts):
                            image_count += 1
                
                if image_count < 6:
                    st.warning(f"Only {image_count} images found. Please upload a dataset with at least 6 images.")
                else:
                    st.success(f"Dataset extracted successfully! Found {image_count} images.")
                    
                    # Option to use directory structure as classes
                    use_dirs = st.checkbox("Use directory structure as ground truth classes", value=True)
                    
                    # Add a button to classify
                    classify_button = st.button("Classify Dataset", type="primary")
                    
                    if classify_button:
                        with st.spinner(f"Classifying {image_count} images..."):
                            # Process the dataset
                            results = []
                            true_labels = []
                            pred_labels = []
                            
                            # First, get all immediate subdirectories as potential classes
                            potential_classes = []
                            for item in os.listdir(tmp_dir):
                                if os.path.isdir(os.path.join(tmp_dir, item)):
                                    potential_classes.append(item)
                            
                            for root, _, files in os.walk(tmp_dir):
                                for file in files:
                                    if any(file.lower().endswith(ext) for ext in image_exts):
                                        file_path = os.path.join(root, file)
                                        try:
                                            # Open and classify the image
                                            img = Image.open(file_path)
                                            class_idx, probabilities = classify_image(img)
                                            
                                            # Get class name and probability
                                            if class_idx < len(class_names):
                                                pred_class = class_names[class_idx]
                                            else:
                                                pred_class = f"Unknown ({class_idx})"
                                            
                                            confidence = probabilities[class_idx] * 100
                                            
                                            # Determine ground truth if using directory structure
                                            true_class = "Unknown"
                                            if use_dirs:
                                                rel_dir = os.path.relpath(root, tmp_dir)
                                                parent_dir = rel_dir.split(os.sep)[0]
                                                if parent_dir in potential_classes:
                                                    true_class = parent_dir
                                            
                                            # Store results
                                            results.append({
                                                "Image": os.path.relpath(file_path, tmp_dir),
                                                "Prediction": pred_class,
                                                "Confidence": confidence,
                                                "True Class": true_class if use_dirs else "N/A"
                                            })
                                            
                                            # Store for metrics if we have ground truth
                                            if use_dirs and true_class in class_names:
                                                true_labels.append(class_names.index(true_class))
                                                pred_labels.append(class_idx)
                                                
                                        except Exception as e:
                                            st.warning(f"Error processing {file}: {e}")
                        
                        # Display summary table
                        st.subheader("Classification Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Display prediction distribution
                        st.subheader("Prediction Distribution")
                        prediction_counts = results_df["Prediction"].value_counts()
                        st.bar_chart(prediction_counts)
                        
                        # If we have ground truth, show metrics
                        if use_dirs and true_labels and pred_labels:
                            st.subheader("Performance Metrics")
                            
                            # Filter to only include classes with examples
                            valid_classes = list(set(true_labels).union(set(pred_labels)))
                            filtered_class_names = [class_names[i] for i in valid_classes if i < len(class_names)]
                            
                            report_dict = classification_report(
                                true_labels, pred_labels, 
                                target_names=filtered_class_names, 
                                output_dict=True
                            )
                            report_df = pd.DataFrame(report_dict).transpose()
                            st.dataframe(report_df.style.format(precision=2))
                            
                            # Confusion matrix
                            cm = confusion_matrix(true_labels, pred_labels)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                            ax.figure.colorbar(im, ax=ax)
                            
                            # Use filtered class names for confusion matrix
                            ax.set(xticks=np.arange(len(filtered_class_names)),
                                  yticks=np.arange(len(filtered_class_names)),
                                  xticklabels=filtered_class_names, 
                                  yticklabels=filtered_class_names,
                                  ylabel='True label',
                                  xlabel='Predicted label')
                            
                            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                            
                            # Add text annotations
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    ax.text(j, i, format(cm[i, j], 'd'),
                                            ha="center", va="center", 
                                            color="white" if cm[i, j] > cm.max() / 2 else "black")
                            fig.tight_layout()
                            st.pyplot(fig)