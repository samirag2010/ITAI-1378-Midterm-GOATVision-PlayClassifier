import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import gdown

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="G.O.A.T Vision", page_icon="⚽", layout="centered")

st.title("G.O.A.T Vision 🐐⚽")
st.write("Soccer Play Classification App")

# -----------------------------
# Classes
# -----------------------------
CLASS_NAMES = [
    "corner_kick",
    "free_kick",
    "penalty_kick",
    "shot",
    "yellow_card"
]

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model path + download
# -----------------------------
MODEL_PATH = "goatvision_model_state_dict.pth"

if not os.path.exists(MODEL_PATH):
    file_id = "14N0ECk_EqcMw6PdO1fr4bH12Hg30vEoY"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Model loader
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Choose a soccer image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    # 👇 REPLACE OLD OUTPUT WITH THIS
    label = CLASS_NAMES[predicted_index].replace("_", " ").title()
    confidence = probabilities[predicted_index].item()

    st.markdown(f"## 🧠 Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2%}")
    st.progress(float(confidence))
    

    st.subheader("Class Probabilities")
    prob_dict = {
        CLASS_NAMES[i]: float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }
    st.bar_chart(prob_dict)
