# streamlit_app.py — Multimodal Soil & Crop Recommendation System
# ResNet-50 + XGBoost + TSACA Fusion + GRN  |  Accuracy: 98.67%
# Run: streamlit run streamlit_app.py

import os, io, json, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="SoilSense — Crop & Soil Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Green agricultural theme ───────────────────────────────────
st.markdown("""
<style>
  /* Hide Streamlit default header/footer */
  #MainMenu, footer, header { visibility: hidden; }

  /* App background */
  .stApp { background: #f0f4e8; }

  /* Top banner */
  .app-header {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
    color: white; padding: 1.4rem 2rem; border-radius: 14px;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .app-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
  .app-header p  { margin: 0; opacity: .8; font-size: .95rem; }
  .badge {
    margin-left: auto; background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3); padding: .35rem 1rem;
    border-radius: 999px; font-size: .8rem; font-weight: 600;
    white-space: nowrap;
  }
  .badge span { color: #a3e635; }

  /* Section headings */
  .section-head {
    font-size: .68rem; font-weight: 700; letter-spacing: 1.2px;
    color: #2d6a4f; text-transform: uppercase; margin: 1.2rem 0 .5rem;
  }

  /* Result cards */
  .soil-card {
    border-radius: 14px; color: white; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .soil-label  { font-size: .65rem; opacity: .75; letter-spacing: 1px; text-transform: uppercase; }
  .soil-name   { font-size: 1.7rem; font-weight: 800; margin: .1rem 0; }
  .soil-conf   { font-size: 1rem; opacity: .9; }

  .crop-card {
    background: white; border: 1.5px solid #d8f3dc; border-radius: 12px;
    padding: .9rem 1rem; margin-bottom: .6rem;
    display: flex; align-items: center; gap: .8rem;
  }
  .crop-card.rank1 { border-color: #52b788; background: #f0faf3; }
  .crop-emoji { font-size: 2rem; line-height: 1; }
  .crop-name  { font-weight: 700; font-size: 1rem; color: #1a1a2e; }
  .crop-fert  { font-size: .75rem; color: #52525b; margin-top: .1rem; }
  .crop-npk   { font-size: .7rem; color: #a1a1aa; }
  .stars      { font-size: .9rem; color: #f59e0b; letter-spacing: 2px; }

  .fert-card {
    background: linear-gradient(135deg, #7c5c3a, #a06840);
    color: white; border-radius: 14px; padding: 1.1rem 1.3rem;
    margin-top: .5rem;
  }
  .fert-label { font-size: .65rem; opacity: .75; letter-spacing: 1px; text-transform: uppercase; }
  .fert-name  { font-size: 1.1rem; font-weight: 700; margin: .15rem 0 .1rem; }
  .fert-npk   { font-size: .85rem; opacity: .85; }

  /* Streamlit widget overrides */
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSelectbox"] label { font-weight: 600; color: #374151; font-size: .85rem; }
  div[data-testid="stFileUploader"] { border: 2px dashed #95d5b2; border-radius: 12px; background: #f0faf3; }

  /* Predict button */
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2d6a4f, #40916c);
    color: white; border: none; border-radius: 10px;
    padding: .7rem 2.5rem; font-size: 1rem; font-weight: 600;
    width: 100%; box-shadow: 0 4px 12px rgba(45,106,79,.35);
    transition: all .2s;
  }
  div[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 18px rgba(45,106,79,.5); transform: translateY(-1px);
  }

  /* Progress bar colour */
  .stProgress > div > div { background: #40916c !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def mpath(n): return os.path.join(BASE, n)

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (identical to app.py / training)
# ══════════════════════════════════════════════════════════════

class ResNet50Classifier(nn.Module):
    def __init__(self, nc, fd=512):
        super().__init__()
        base = models.resnet50(weights=None)
        self.backbone   = nn.Sequential(*list(base.children())[:-2])
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, fd),
            nn.BatchNorm1d(fd),   nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(fd, nc))
    def forward(self, x, return_features=False):
        f = self.projection(self.pool(self.backbone(x)))
        if return_features: return f
        return self.classifier(f)


class TabProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class TSACAFusion(nn.Module):
    def __init__(self, img_dim, tab_dim, fd, nh, nl=3):
        super().__init__()
        self.ip = nn.Sequential(nn.Linear(img_dim, fd), nn.LayerNorm(fd), nn.ReLU())
        self.tp = nn.Sequential(nn.Linear(tab_dim, fd), nn.LayerNorm(fd), nn.ReLU())
        self.ca = nn.ModuleList([
            nn.MultiheadAttention(fd, nh, dropout=0.1, batch_first=True)
            for _ in range(nl)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(fd, fd*4), nn.GELU(),
                          nn.Dropout(0.1), nn.Linear(fd*4, fd))
            for _ in range(nl)])
        self.nm   = nn.ModuleList([nn.LayerNorm(fd) for _ in range(nl*2)])
        self.gate = nn.Sequential(
            nn.Linear(fd*2, fd*2), nn.ReLU(),
            nn.Linear(fd*2, fd),   nn.Sigmoid())
        self.out  = nn.Sequential(nn.Linear(fd, fd), nn.LayerNorm(fd), nn.ReLU())
    def forward(self, img_f, tab_f):
        ip = self.ip(img_f).unsqueeze(1)
        tp = self.tp(tab_f).unsqueeze(1)
        x  = tp
        for i, (a, f) in enumerate(zip(self.ca, self.ff)):
            ao, _ = a(query=x, key=ip, value=ip)
            x = self.nm[i*2](x + ao)
            x = self.nm[i*2+1](x + f(x))
        x  = x.squeeze(1)
        gw = self.gate(torch.cat([x, tp.squeeze(1)], dim=-1))
        return self.out(gw * x + (1 - gw) * tp.squeeze(1))


class GatedLinearUnit(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o); self.gate = nn.Linear(i, o)
    def forward(self, x): return self.fc(x) * torch.sigmoid(self.gate(x))


class GRNBlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim*2); self.elu = nn.ELU()
        self.fc2  = nn.Linear(dim*2, dim)
        self.glu  = GatedLinearUnit(dim, dim)
        self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(drop)
    def forward(self, x):
        h = self.elu(self.fc1(x)); h = self.drop(self.fc2(h))
        return self.norm(self.glu(h) + x)


class GRNCropPredictor(nn.Module):
    def __init__(self, in_dim, nc, nb=5, drop=0.2):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.ReLU())
        self.blocks = nn.ModuleList([GRNBlock(in_dim, drop) for _ in range(nb)])
        self.head   = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(drop / 2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, nc))
    def forward(self, f):
        x = self.proj(f)
        for b in self.blocks: x = b(x)
        logits = self.head(x)
        return logits, torch.softmax(logits, -1).max(-1).values


class FusionGRNModel(nn.Module):
    def __init__(self, img_dim, xgb_dim, fused_dim, num_heads, num_classes):
        super().__init__()
        self.tsaca = TSACAFusion(img_dim, xgb_dim, fused_dim, num_heads)
        self.grn   = GRNCropPredictor(fused_dim, num_classes)
    def forward(self, img_f, tab_f):
        return self.grn(self.tsaca(img_f, tab_f))


# ══════════════════════════════════════════════════════════════
# CACHED MODEL LOADING — runs only once per session
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_everything():
    with open(mpath("model_config.json")) as f: cfg = json.load(f)
    with open(mpath("class_names.json"))  as f: cls = json.load(f)

    img_dim   = cfg["IMG_FEAT_DIM"]   # 512
    xgb_dim   = cfg["XGB_PROJ_DIM"]   # 256
    fused_dim = cfg["FUSED_DIM"]      # 512
    num_heads = cfg["NUM_HEADS"]      # 8
    num_cls   = cfg["NUM_CLASSES"]    # 6
    tab_dim   = cfg["TAB_FEAT_DIM"]   # 19
    num_cols  = cfg["NUMERIC_COLS"]
    img_size  = cfg["IMG_SIZE"]       # 224

    img_m  = ResNet50Classifier(num_cls, img_dim)
    tab_p  = TabProjector(tab_dim, xgb_dim)
    fusion = FusionGRNModel(img_dim, xgb_dim, fused_dim, num_heads, num_cls)

    img_m.load_state_dict( torch.load(mpath("img_model.pt"),     map_location="cpu", weights_only=True))
    tab_p.load_state_dict( torch.load(mpath("tab_projector.pt"), map_location="cpu", weights_only=True))
    fusion.load_state_dict(torch.load(mpath("fusion_model.pt"),  map_location="cpu", weights_only=True))

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(mpath("xgb_model.json"))

    with open(mpath("scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    img_m.eval(); tab_p.eval(); fusion.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return img_m, tab_p, fusion, xgb_clf, scaler, cls, num_cols, tf


# ── Lookup maps ────────────────────────────────────────────────
SEASON_MAP = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
IRRIG_MAP  = {"Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3}
PREV_MAP   = {"Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3,
               "Sugarcane": 4, "Tomato": 5, "Wheat": 6}
REGION_MAP = {"Central": 0, "East": 1, "North": 2, "South": 3, "West": 4}

SOIL_FERT_MAP = {
    "Alluvial Soil": {"fertilizer": "NPK 20:20:0 + Zinc",  "npk": "N:P:K = 80:40:20 kg/ha"},
    "Black Soil":    {"fertilizer": "Urea + MOP",           "npk": "N:P:K = 60:30:30 kg/ha"},
    "Clay Soil":     {"fertilizer": "Urea + DAP",           "npk": "N:P:K = 60:60:60 kg/ha"},
    "Laterite Soil": {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:60:20 kg/ha"},
    "Red Soil":      {"fertilizer": "NPK 17:17:17",         "npk": "N:P:K = 50:50:50 kg/ha"},
    "Yellow Soil":   {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:30:20 kg/ha"},
}

CROP_FERT_MAP = {
    "Cotton":    {"fertilizer": "NPK 17:17:17",  "npk": "50:50:50 kg/ha"},
    "Maize":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
    "Potato":    {"fertilizer": "NPK 15:15:15",  "npk": "180:120:80 kg/ha"},
    "Rice":      {"fertilizer": "Urea + SSP",    "npk": "100:50:25 kg/ha"},
    "Sugarcane": {"fertilizer": "NPK 20:10:10",  "npk": "250:85:115 kg/ha"},
    "Tomato":    {"fertilizer": "NPK 12:32:16",  "npk": "200:150:200 kg/ha"},
    "Wheat":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
}

CROP_MAP = {
    ("Red Soil",      "Kharif"): ["Cotton",    "Maize",      "Groundnut",  "Tomato"],
    ("Red Soil",      "Rabi")  : ["Wheat",     "Sunflower",  "Linseed",    "Potato"],
    ("Red Soil",      "Zaid")  : ["Watermelon","Cucumber",   "Bitter Gourd","Moong"],
    ("Alluvial Soil", "Kharif"): ["Rice",      "Sugarcane",  "Maize",      "Jute"],
    ("Alluvial Soil", "Rabi")  : ["Wheat",     "Mustard",    "Barley",     "Peas"],
    ("Alluvial Soil", "Zaid")  : ["Watermelon","Muskmelon",  "Cucumber",   "Moong"],
    ("Black Soil",    "Kharif"): ["Cotton",    "Sorghum",    "Soybean",    "Groundnut"],
    ("Black Soil",    "Rabi")  : ["Wheat",     "Chickpea",   "Linseed",    "Safflower"],
    ("Black Soil",    "Zaid")  : ["Sunflower", "Sesame",     "Maize",      "Moong"],
    ("Clay Soil",     "Kharif"): ["Rice",      "Jute",       "Sugarcane",  "Taro"],
    ("Clay Soil",     "Rabi")  : ["Wheat",     "Barley",     "Mustard",    "Spinach"],
    ("Clay Soil",     "Zaid")  : ["Cucumber",  "Bitter Gourd","Pumpkin",   "Moong"],
    ("Laterite Soil", "Kharif"): ["Cashew",    "Rubber",     "Tea",        "Coffee"],
    ("Laterite Soil", "Rabi")  : ["Tapioca",   "Groundnut",  "Turmeric",   "Ginger"],
    ("Laterite Soil", "Zaid")  : ["Mango",     "Pineapple",  "Jackfruit",  "Banana"],
    ("Yellow Soil",   "Kharif"): ["Rice",      "Maize",      "Groundnut",  "Sesame"],
    ("Yellow Soil",   "Rabi")  : ["Wheat",     "Mustard",    "Potato",     "Barley"],
    ("Yellow Soil",   "Zaid")  : ["Sunflower", "Moong",      "Cucumber",   "Tomato"],
}

SOIL_COLOR = {
    "Alluvial Soil": "linear-gradient(135deg,#7c6f52,#a08c6a)",
    "Black Soil":    "linear-gradient(135deg,#2d2d2d,#4a4a4a)",
    "Clay Soil":     "linear-gradient(135deg,#7a4f38,#a06848)",
    "Laterite Soil": "linear-gradient(135deg,#8b3a20,#b05030)",
    "Red Soil":      "linear-gradient(135deg,#9b1c1c,#c53030)",
    "Yellow Soil":   "linear-gradient(135deg,#856404,#b28a0a)",
}

SOIL_ICON = {
    "Alluvial Soil": "🏜️", "Black Soil": "⬛", "Clay Soil": "🧱",
    "Laterite Soil": "🪨", "Red Soil": "🔴", "Yellow Soil": "🟡",
}

CROP_EMOJI = {
    "Rice": "🌾", "Wheat": "🌾", "Cotton": "🌿", "Maize": "🌽",
    "Sugarcane": "🎋", "Tomato": "🍅", "Potato": "🥔", "Groundnut": "🥜",
    "Jute": "🌿", "Barley": "🌾", "Mustard": "🌻", "Peas": "🫛",
    "Sorghum": "🌾", "Soybean": "🌱", "Sunflower": "🌻", "Chickpea": "🫘",
    "Linseed": "🌼", "Safflower": "🌸", "Sesame": "🌿", "Moong": "🫘",
    "Cashew": "🥜", "Rubber": "🌳", "Tea": "🍵", "Coffee": "☕",
    "Tapioca": "🥔", "Turmeric": "🟡", "Ginger": "🫚",
    "Watermelon": "🍉", "Muskmelon": "🍈", "Cucumber": "🥒",
    "Bitter Gourd": "🥬", "Pumpkin": "🎃", "Taro": "🌱",
    "Spinach": "🥬", "Mango": "🥭", "Pineapple": "🍍",
    "Jackfruit": "🟡", "Banana": "🍌",
}


# ══════════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ══════════════════════════════════════════════════════════════

def run_inference(img_model, tab_proj, fusion, xgb_clf, scaler,
                  class_names, num_cols, tf,
                  pil_img, n, p, k, temp, hum, rain, ph, yld, fert,
                  season, irrig, prev, region):

    # Tabular features
    num_raw = np.array([[n, p, k, temp, hum, rain, ph, yld, fert]])
    num_sc  = scaler.transform(pd.DataFrame(num_raw, columns=num_cols))
    cat_enc = np.array([[
        SEASON_MAP[season], IRRIG_MAP[irrig],
        PREV_MAP[prev],     REGION_MAP[region],
    ]])
    scaled_feat = np.concatenate([num_sc, cat_enc], axis=1).astype(np.float32)

    xgb_probs = xgb_clf.predict_proba(scaled_feat)           # (1, 6)
    tab_raw   = np.concatenate([xgb_probs, scaled_feat], axis=1).astype(np.float32)
    tab_t     = torch.tensor(tab_raw, dtype=torch.float32)

    # Image
    img_t = tf(pil_img.convert("RGB")).unsqueeze(0)

    # Inference
    with torch.no_grad():
        img_feat  = img_model(img_t, return_features=True)
        tab_feat  = tab_proj(tab_t)
        logits, _ = fusion(img_feat, tab_feat)

    fusion_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Calibrated ensemble
    xgb_p   = xgb_probs[0]
    top2    = np.partition(fusion_probs, -2)[-2:]
    gap     = float(top2[-1] - top2[-2])
    xgb_w   = 0.45 if gap < 0.20 else 0.30
    blended = (1 - xgb_w) * fusion_probs + xgb_w * xgb_p

    # Red ↔ Yellow calibration
    RED_IDX    = class_names.index("Red Soil")
    YELLOW_IDX = class_names.index("Yellow Soil")
    if blended[RED_IDX] > 0.30 or blended[YELLOW_IDX] > 0.30:
        ph_score     = max(0.0, (6.5 - ph) / 6.5)
        k_score      = max(0.0, (50.0 - k) / 50.0)
        yellow_boost = 0.12 * (ph_score + k_score) / 2.0
        blended[YELLOW_IDX] = min(1.0, blended[YELLOW_IDX] + yellow_boost)
        blended[RED_IDX]    = max(0.0, blended[RED_IDX]    - yellow_boost)
        blended = blended / blended.sum()

    pred_idx   = int(np.argmax(blended))
    soil_name  = class_names[pred_idx]
    confidence = round(float(blended[pred_idx]) * 100, 2)
    all_probs  = {class_names[i]: round(float(blended[i]) * 100, 2)
                  for i in range(len(class_names))}

    soil_fert = SOIL_FERT_MAP.get(soil_name,
                {"fertilizer": "NPK 14:14:14", "npk": "N:P:K = 60:30:30 kg/ha"})

    crops_all = CROP_MAP.get(
        (soil_name, season),
        CROP_MAP.get((soil_name, "Kharif"), ["Wheat", "Rice", "Maize"]))

    crop_recs = []
    for i, crop in enumerate(crops_all[:3]):
        cf = CROP_FERT_MAP.get(crop, {"fertilizer": "NPK 14:14:14", "npk": "60:40:20 kg/ha"})
        crop_recs.append({"name": crop, "rank": i+1, "stars": 5-i,
                           "fertilizer": cf["fertilizer"], "npk": cf["npk"]})

    return soil_name, confidence, all_probs, soil_fert, crop_recs


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <span style="font-size:2.5rem">🌱</span>
  <div>
    <h1>SoilSense</h1>
    <p>Multimodal Crop &amp; Soil Recommendation System</p>
  </div>
  <div class="badge">Accuracy: <span>98.67%</span></div>
</div>
""", unsafe_allow_html=True)

# Load models
img_model, tab_proj, fusion, xgb_clf, scaler, CLASS_NAMES, NUMERIC_COLS, eval_tf = load_everything()

# ── Two-column layout ──────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-head">📷 Soil Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded, use_container_width=True, caption="Uploaded soil image")

    st.markdown('<div class="section-head">🧪 Soil & Climate Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        n    = st.number_input("Nitrogen (N)",       0.0, 200.0, 90.0,  step=1.0, help="kg/ha")
        temp = st.number_input("Temperature (°C)",   0.0,  50.0, 25.0,  step=0.1)
        ph   = st.number_input("Soil pH",            3.0,  10.0,  6.5,  step=0.01)
    with c2:
        p    = st.number_input("Phosphorus (P)",     0.0, 200.0, 42.0,  step=1.0, help="kg/ha")
        hum  = st.number_input("Humidity (%)",       0.0, 100.0, 80.0,  step=1.0)
        yld  = st.number_input("Yield Last Season",  0.0,15000.0,2500.0,step=10.0, help="kg/ha")
    with c3:
        k    = st.number_input("Potassium (K)",      0.0, 200.0, 43.0,  step=1.0, help="kg/ha")
        rain = st.number_input("Rainfall (mm)",      0.0,3000.0, 200.0, step=5.0)
        fert = st.number_input("Fertilizer Used",    0.0,1000.0, 120.0, step=5.0, help="kg/ha last season")

    st.markdown('<div class="section-head">🌾 Farm Context</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        prev   = st.selectbox("Previous Crop",
                              ["Wheat","Rice","Maize","Cotton","Potato","Sugarcane","Tomato"])
    with d2:
        irrig  = st.selectbox("Irrigation Type", ["Canal","Drip","Rainfed","Sprinkler"])
        region = st.selectbox("Region", ["South","North","East","West","Central"])

    predict_clicked = st.button("🔍  Analyze Soil & Get Recommendations")

# ── Results column ─────────────────────────────────────────────
with right:
    st.markdown('<div class="section-head">📊 Analysis Results</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;color:#a1a1aa;">
          <div style="font-size:3.5rem;margin-bottom:1rem;opacity:.4">🌾</div>
          <p>Upload a soil image and fill in the parameters,<br>
             then click <strong>Analyze</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded is None:
        st.error("Please upload a soil image before analyzing.")

    else:
        with st.spinner("Running AI model inference…"):
            try:
                pil_img = Image.open(uploaded)
                soil_name, confidence, all_probs, soil_fert, crop_recs = run_inference(
                    img_model, tab_proj, fusion, xgb_clf, scaler,
                    CLASS_NAMES, NUMERIC_COLS, eval_tf,
                    pil_img, n, p, k, temp, hum, rain, ph, yld, fert,
                    season, irrig, prev, region,
                )

                # ── Soil type card ──────────────────────────────
                grad = SOIL_COLOR.get(soil_name, "linear-gradient(135deg,#2d6a4f,#40916c)")
                icon = SOIL_ICON.get(soil_name, "🪨")
                st.markdown(f"""
                <div class="soil-card" style="background:{grad}">
                  <div style="display:flex;align-items:center;gap:1rem">
                    <span style="font-size:2.8rem">{icon}</span>
                    <div style="flex:1">
                      <div class="soil-label">Detected Soil Type</div>
                      <div class="soil-name">{soil_name}</div>
                    </div>
                    <div style="text-align:center;background:rgba(255,255,255,.18);
                                border-radius:12px;padding:.6rem 1.1rem">
                      <div style="font-size:1.6rem;font-weight:800">{confidence}%</div>
                      <div style="font-size:.65rem;opacity:.8">confidence</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── All probabilities ───────────────────────────
                st.markdown("**All Soil Probabilities**")
                for name, pct in sorted(all_probs.items(), key=lambda x: -x[1]):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(int(pct), text=name)
                    with col_b:
                        st.markdown(f"<div style='text-align:right;font-weight:600;padding-top:.4rem'>{pct}%</div>",
                                    unsafe_allow_html=True)

                # ── Top 3 recommended crops ─────────────────────
                st.markdown("**Top Recommended Crops**")
                for crop in crop_recs:
                    emoji  = CROP_EMOJI.get(crop["name"], "🌱")
                    stars  = "★" * crop["stars"] + "☆" * (5 - crop["stars"])
                    border = "#52b788" if crop["rank"] == 1 else "#d8f3dc"
                    bg     = "#f0faf3" if crop["rank"] == 1 else "white"
                    st.markdown(f"""
                    <div class="crop-card" style="border-color:{border};background:{bg}">
                      <span class="crop-emoji">{emoji}</span>
                      <div style="flex:1">
                        <div class="crop-name">{crop["name"]}</div>
                        <div class="crop-fert">🌿 {crop["fertilizer"]}</div>
                        <div class="crop-npk">NPK: {crop["npk"]}</div>
                      </div>
                      <div style="text-align:center">
                        <div class="stars">{stars}</div>
                        <div style="font-size:.62rem;color:#a1a1aa;font-weight:600">RANK #{crop["rank"]}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Soil fertilizer card ────────────────────────
                st.markdown(f"""
                <div class="fert-card">
                  <div style="display:flex;align-items:flex-start;gap:.8rem">
                    <span style="font-size:2rem">🌿</span>
                    <div>
                      <div class="fert-label">Soil Fertilizer Recommendation</div>
                      <div class="fert-name">{soil_fert["fertilizer"]}</div>
                      <div class="fert-npk">{soil_fert["npk"]}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
