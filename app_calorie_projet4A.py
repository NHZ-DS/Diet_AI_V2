import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
import base64
import joblib

# --- 1. REDÉFINITION DE LA CLASSE (Indispensable pour charger les poids) ---
class CalorieRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=None)
        n_inputs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x): 
        return self.base_model(x)

# --- 2. CONFIGURATION ET CHARGEMENT ---
scaler_path = "calorie_scaler.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("Fichier 'calorie_scaler.joblib' introuvable !")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="CalorieScan AI", page_icon="🍎", layout="centered")

# --- 3. STYLISATION (CSS) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

def set_background(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; background-attachment: fixed; }}
        .main .block-container {{ background-color: rgba(0, 0, 0, 0.6); padding: 40px; border-radius: 25px; color: white; }}
        h1, h3, p, label, span {{ color: white !important; }}
        div.stButton > button {{ width: 100%; border-radius: 15px; background-color: #2ECC71; color: white; font-weight: bold; height: 3em; border: none; }}
        .result-card {{ background-color: rgba(255, 255, 255, 0.15); padding: 20px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.2); text-align: center; }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

img_fond = "fond_decran.png"
if os.path.exists(img_fond):
    set_background(img_fond)

# --- 4. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_trained_model():
    model_path = "best_calorie_model.pth"
    if os.path.exists(model_path):
        try:
            # On utilise la classe définie plus haut
            model = CalorieRegressor()
            
            # Chargement intelligent du state_dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Si le checkpoint contient un dictionnaire 'state_dict' (cas fréquent)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Erreur lors du chargement des poids : {e}")
    else:
        st.error("Fichier 'best_calorie_model.pth' introuvable !")
    return None

model = load_trained_model()

# --- 5. INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🍎 CalorieScan AI</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📸 Prendre une photo", "📁 Importer une image"])

img_file = None
with tab1:
    img_file_cam = st.camera_input("Capturez votre plat")
    if img_file_cam:
        img_file = img_file_cam
with tab2:
    img_file_up = st.file_uploader("Choisissez une photo...", type=["jpg", "jpeg", "png"])
    if img_file_up:
        img_file = img_file_up

# --- 6. LOGIQUE D'ANALYSE ---
if img_file:
    if not hasattr(img_file, 'type') or img_file.type != "camera":
         st.image(img_file, caption="Image importée", use_container_width=True)

    image = Image.open(img_file).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    if model is not None:
        with st.spinner('Analyse en cours...'):
            with torch.no_grad():
                output = model(input_tensor)
                output_np = output.cpu().numpy()
                # Correction du format pour le scaler
                real_calories = scaler.inverse_transform(output_np.reshape(-1, 1))
                kcal = max(0, float(real_calories[0][0]))

        st.markdown("<h3 style='text-align: center;'>🍽️ Résultats</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:32px; font-weight:bold; color:#FF5E5E !important; margin:0;">{kcal:.0f}</p>
            <p style="font-size:14px; color:white !important; letter-spacing: 2px;">CALORIES ESTIMÉES (KCAL)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Nouvelle analyse"):
            st.rerun()

st.markdown("<br><p style='text-align: center; font-size: 10px; color: #D5D8DC !important;'>🎓 Projet Académique - 2026</p>", unsafe_allow_html=True)