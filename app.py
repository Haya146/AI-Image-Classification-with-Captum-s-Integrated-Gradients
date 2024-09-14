import streamlit as st
import torch
import numpy as np
import cv2
import pandas as pd
from captum.attr import IntegratedGradients, visualization as viz
from torchvision import models, transforms
from PIL import Image
import json
import requests

# Load the ResNet50 model pre-trained on ImageNet (in PyTorch)
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Load ImageNet labels to map label numbers to class names
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels_map = json.loads(response.text)

# Function to preprocess image for the model (PyTorch)
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Convert image to PIL format for prediction
def decode_predictions(preds, top=5):
    decoded = torch.nn.functional.softmax(preds, dim=1)
    top_probs, top_labels = torch.topk(decoded, top)
    return top_probs, top_labels

# Function to get ImageNet class names from label numbers
def get_class_names(top_labels):
    return [labels_map[i] for i in top_labels]

# Function to generate Integrated Gradients using Captum
def generate_integrated_gradients(image, model, target_class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(image)  # Black image as the baseline
    
    # Instantiate IntegratedGradients object
    ig = IntegratedGradients(model)
    
    # Calculate the integrated gradients
    attributions = ig.attribute(image, target=target_class_idx, baselines=baseline, n_steps=steps)
    return attributions

# Function to visualize attributions using Captum's viz utilities
def visualize_attributions(attributions, processed_img):
    # Remove batch dimension and convert tensors to NumPy arrays
    feature_imp = attributions[0].detach().numpy()  # Detach before calling .numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)  # Convert from (3, 224, 224) to (224, 224, 3)
    
    processed_img_np = processed_img.detach().numpy().transpose(1, 2, 0)  # Convert to (224, 224, 3)
    
    # Visualize using Captum's built-in visualization method
    fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6, 6))
    return fig

# Display the heatmap over the original image
def overlay_heatmap(image, attributions):
    attributions = attributions.squeeze().cpu().detach().numpy().sum(axis=0)  # Sum over the color channels
    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))  # Normalize to [0, 1]
    heatmap = cv2.resize(attributions, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = heatmap * 0.4 + image
    return overlayed_image

# Streamlit UI
st.title("AI Image Classification with Captum's Integrated Gradients")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and make predictions using PyTorch model
    processed_image = preprocess_image(image)
    processed_image.requires_grad = True
    preds = model(processed_image)
    
    # Decode the top 5 predictions
    top_probs, top_labels = decode_predictions(preds, top=5)
    
    # Get the class names (animal names) for the top predictions
    top_class_names = get_class_names(top_labels[0].cpu().numpy())
    
    # Display the top 5 predicted categories with class names
    st.subheader("Top 5 Predictions:")
    labels = []
    scores = []
    for i in range(top_probs.size(1)):
        class_name = top_class_names[i]
        score = top_probs[0][i].item()
        st.write(f"{i+1}. {class_name}: {score:.4f}")
        labels.append(class_name)
        scores.append(score)

    # Display top 5 predictions in a bar chart
    st.subheader("Bar Chart of Top 5 Predictions:")
    df = pd.DataFrame({"Label": labels, "Score": scores})
    st.bar_chart(df.set_index("Label"))
    
    # Generate Integrated Gradients heatmap for the top prediction
    top_class_idx = top_labels[0][0].item()
    attributions = generate_integrated_gradients(processed_image, model, target_class_idx=top_class_idx)
    
    # Visualize attributions using Captum
    st.subheader("Captum Integrated Gradients Visualization:")
    fig = visualize_attributions(attributions, processed_image[0])
    st.pyplot(fig)
    
    # Overlay heatmap over the original image (optional visualization)
    overlayed_image = overlay_heatmap(image, attributions)
    
    # Normalize the image to be within the range [0, 1]
    normalized_overlayed_image = overlayed_image / 255.0
    normalized_overlayed_image = np.clip(normalized_overlayed_image, 0, 1)

    # Display the heatmap overlaid on the image
    st.subheader("Integrated Gradients Heatmap for the Top Prediction:")
    st.image(normalized_overlayed_image, caption="Integrated Gradients Heatmap", use_column_width=True)
