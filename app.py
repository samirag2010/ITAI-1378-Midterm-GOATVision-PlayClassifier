import os
import gdown
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="G.O.A.T Vision",
    page_icon="⚽",
    layout="centered"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title-text {
            font-size: 3rem;
            font-weight: 800;
            color: #1f3b73;
            margin-bottom: 0.2rem;
        }
        .subtitle-text {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .prediction-box {
            background: linear-gradient(135deg, #e8f1ff, #f4f8ff);
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid #dbeafe;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .prediction-label {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
        }
        .confidence-text {
            font-size: 1rem;
            color: #334155;
            margin-top: 0.4rem;
        }
        .small-note {
            font-size: 0.9rem;
            color: #64748b;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title-text">G.O.A.T Vision 🐐⚽</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Game Outcome Action Tracker for Soccer Play Classification</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.write(
    "Upload a soccer image and G.O.A.T Vision will classify the play type using a trained ResNet50 model."
)
st.sidebar.markdown("### Classes")
st.sidebar.write(
    "- Corner Kick\n"
    "- Free Kick\n"
    "- Penalty Kick\n"
    "- Shot\n"
    "- Yellow Card"
)

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

DISPLAY_NAMES = {
    "corner_kick": "Corner Kick",
    "free_kick": "Free Kick",
    "penalty_kick": "Penalty Kick",
    "shot": "Shot",
    "yellow_card": "Yellow Card"
}

EMOJI_MAP = {
    "corner_kick": "🚩",
    "free_kick": "🎯",
    "penalty_kick": "🥅",
    "shot": "⚽",
    "yellow_card": "🟨"
}

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
# Upload section
# -----------------------------
st.markdown("### Upload a soccer image")
uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    raw_label = CLASS_NAMES[predicted_index]
    label = DISPLAY_NAMES[raw_label]
    emoji = EMOJI_MAP[raw_label]
    confidence = probabilities[predicted_index].item()

    with col2:
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">{emoji} {label}</div>
                <div class="confidence-text">Confidence: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(confidence))
        st.markdown(
            '<div class="small-note">Confidence reflects how strongly the model favors this class over the others.</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Class Probabilities")
    prob_dict = {
        DISPLAY_NAMES[CLASS_NAMES[i]]: float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }
    st.bar_chart(prob_dict)

else:
    st.info("Upload a soccer image to get a prediction.")
