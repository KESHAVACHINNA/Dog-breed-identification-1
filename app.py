import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

# Title
st.title("🐶 Dog Breed Identification App")

# Load Model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_classes = 120  # Adjust this if needed based on your dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load label names (if you have a labels.txt or define manually)
labels = [f"Breed {i}" for i in range(120)]  # Dummy labels
# If you have labels in a file:
# with open('labels.txt', 'r') as f:
#     labels = f.read().splitlines()

# Upload image
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        st.success(f"Predicted Breed: **{labels[predicted.item()]}**")
