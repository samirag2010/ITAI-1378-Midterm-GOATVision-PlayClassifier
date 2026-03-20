import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Title
st.title("G.O.A.T Vision 🐐⚽")
st.write("Soccer Play Classification App")

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=False))
model.eval()

# Classes
classes = ['corner_kick', 'free_kick', 'penalty_kick', 'shot', 'yellow_card']

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Upload image
uploaded_file = st.file_uploader("Upload a soccer image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    st.success(f"Prediction: {classes[pred.item()]}")
