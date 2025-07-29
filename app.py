import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load the trained model (adjust the path if needed)
model = torch.load('model.pth', map_location='cpu')
model.eval()

# Example breed list – replace with your actual classes
breed_names = ['beagle', 'golden_retriever', 'labrador', 'poodle', 'pug', 'shih_tzu', 'german_shepherd']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("🐶 Dog Breed Identifier")
uploaded = st.file_uploader("Upload an image of a dog", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
        breed = breed_names[prediction.item()]
    
    st.success(f"Predicted Breed: **{breed}**")
