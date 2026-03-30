# streamlit_app.py — Multimodal Soil & Crop Recommendation System
# ResNet-50 + XGBoost + TSACA Fusion + GRN  |  Accuracy: 98.67%
# Run: streamlit run streamlit_app.py

import io, os, json, pickle
import requests
import numpy as np, pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Crop & Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme state ────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# ── Theme CSS definitions ───────────────────────────────────────
_BASE_CSS = """
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 2rem 2rem; max-width: 1200px; }
  h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
  .hero h1, .hero p { color: #ffffff !important; }
  .section-card { background: var(--section-bg); padding: 16px; border-radius: 12px;
                  border-left: 4px solid var(--accent); margin-bottom: 16px; }
  .section-title { color: var(--section-title); margin: 0 0 10px 0; }
  div.stButton > button {
    background: #2E7D32; color: white; border: none;
    padding: 14px 48px; font-size: 18px; font-weight: 600;
    border-radius: 30px; width: 100%; cursor: pointer;
    margin-top: 16px; transition: background 0.2s;
    white-space: nowrap !important;
  }
  div.stButton > button:hover { background: #1B5E20; }
  /* Theme toggle button — override width/padding for compact size */
  div[data-testid="column"]:last-child div.stButton > button {
    min-width: 90px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    width: auto !important;
    margin-top: 0 !important;
    white-space: nowrap !important;
    text-align: left !important;
  }
  @media (max-width: 768px) {
    div[data-testid="column"]:last-child {
      display: flex;
      justify-content: flex-start;
    }
    div[data-testid="column"]:last-child div.stButton {
      width: 100%;
    }
  }
</style>
"""

LIGHT_THEME = _BASE_CSS + """
<style>
  :root { --card-bg: #FFFFFF; --card-border: #E8F5E9; --text: #212121;
          --subtext: #555; --page-bg: #F9FBF9; --section-bg: #E8F5E9;
          --section-title: #1B5E20; --accent: #2E7D32;
          --prob-track: #f0f0f0; --prob-label: #333; --fert-bg: #FFF8E1; }
  .main, .stApp { background: #F9FBF9 !important; }
  h1,h2,h3,h4,p { color: #212121; }
  .stSelectbox label, .stNumberInput label { color: #1B5E20 !important; font-weight: 500; }
  div[data-testid="stFileUploader"] {
    border: 2px dashed #2E7D32; border-radius: 12px;
    background: #F1F8E9; padding: 8px;
  }
</style>
"""

DARK_THEME = _BASE_CSS + """
<style>
  :root { --card-bg: #1E1E1E; --card-border: #2E2E2E; --text: #FFFFFF;
          --subtext: #AAAAAA; --page-bg: #121212; --section-bg: #1A2A1A;
          --section-title: #E7F7E7; --accent: #81C784;
          --prob-track: #2C2C2C; --prob-label: #CCCCCC; --fert-bg: #2A2000; }
  .main, .stApp { background: #121212 !important; }
  h1,h2,h3,h4,p,label,span { color: #FFFFFF !important; }
  .stSelectbox label, .stNumberInput label { color: #81C784 !important; font-weight: 500; }
  div[data-testid="stFileUploader"] {
    border: 2px dashed #4CAF50; border-radius: 12px;
    background: #1A2A1A; padding: 8px;
  }
  div[data-testid="stNumberInput"] input,
  .stTextInput input { background: #2C2C2C !important; color: #FFFFFF !important; }
  .stSelectbox > div > div { background: #2C2C2C !important; color: #FFFFFF !important; }
  div[data-testid="stExpander"] { background: #1E1E1E !important; }
  .stSlider { filter: brightness(1.4); }
</style>
"""

# Apply theme
if st.session_state.theme == "dark":
    st.markdown(DARK_THEME, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME, unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def mpath(n): return os.path.join(BASE, n)

# ── Model file verification (runs at startup, visible in logs) ──
_MODEL_FILES = [
    "img_model.pt", "fusion_model.pt",
    "tab_projector.pt", "xgb_model.json", "scaler.pkl",
]
_file_status = {}
for _f in _MODEL_FILES:
    _p = mpath(_f)
    _exists = os.path.exists(_p)
    _size   = os.path.getsize(_p) if _exists else 0
    _file_status[_f] = {"exists": _exists, "mb": round(_size / 1024 / 1024, 1)}
    print(f"{_f}: exists={_exists}, size={_size/1024/1024:.1f}MB")

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — identical to training
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
            nn.BatchNorm1d(fd), nn.ReLU(),
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
        self.fc1 = nn.Linear(dim, dim*2); self.elu = nn.ELU()
        self.fc2 = nn.Linear(dim*2, dim)
        self.glu = GatedLinearUnit(dim, dim)
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
# MODEL LOADING — cached per session, loads once
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_all_models():
    """Load and cache all models once per session.
    Raises a clear error if any model file is missing or too small
    (which happens when Git LFS pointers aren't resolved on the server).
    """
    # ── Validate model files before loading ────────────────────
    MIN_SIZES = {
        "img_model.pt":     50,   # expect ~105 MB
        "fusion_model.pt":  20,   # expect ~80 MB
        "tab_projector.pt":  0.5, # expect ~1.6 MB
    }
    for fname, min_mb in MIN_SIZES.items():
        info = _file_status.get(fname, {})
        if not info.get("exists"):
            raise FileNotFoundError(
                f"{fname} not found. Check repository LFS setup.")
        if info["mb"] < min_mb:
            raise ValueError(
                f"{fname} is only {info['mb']:.1f} MB — "
                f"expected >{min_mb} MB. "
                f"Git LFS pointers may not have been resolved on this server. "
                f"Run: git lfs pull")

    with open(mpath("model_config.json")) as f: cfg = json.load(f)
    with open(mpath("class_names.json"))  as f: cls = json.load(f)

    img_dim   = cfg["IMG_FEAT_DIM"]
    xgb_dim   = cfg["XGB_PROJ_DIM"]
    fused_dim = cfg["FUSED_DIM"]
    num_heads = cfg["NUM_HEADS"]
    num_cls   = cfg["NUM_CLASSES"]
    tab_dim   = cfg["TAB_FEAT_DIM"]
    num_cols  = cfg["NUMERIC_COLS"]

    img_m  = ResNet50Classifier(num_cls, img_dim)
    tab_p  = TabProjector(tab_dim, xgb_dim)
    fusion = FusionGRNModel(img_dim, xgb_dim, fused_dim, num_heads, num_cls)

    img_m.load_state_dict(
        torch.load(mpath("img_model.pt"),     map_location="cpu", weights_only=True))
    tab_p.load_state_dict(
        torch.load(mpath("tab_projector.pt"), map_location="cpu", weights_only=True))
    fusion.load_state_dict(
        torch.load(mpath("fusion_model.pt"),  map_location="cpu", weights_only=True))

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(mpath("xgb_model.json"))

    with open(mpath("scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    # Explicit eval() — essential for BatchNorm/Dropout at inference
    img_m.eval(); tab_p.eval(); fusion.eval()

    return img_m, tab_p, fusion, xgb_clf, scaler, cls, num_cols


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


# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════

def run_inference(img_model, tab_proj, fusion, xgb_clf, scaler,
                  class_names, num_cols,
                  img_bytes, n, p, k, temp, hum, rain, ph, yld, fert,
                  season, irrig, prev, region):

    # ── Transform created fresh every call (never cached) ──────
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Image: always decode from raw bytes ────────────────────
    # UploadedFile stream is consumed by st.image(); using saved bytes
    # guarantees a correct, fresh tensor regardless of Streamlit rerenders.
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t   = tf(pil_img).unsqueeze(0)           # (1, 3, 224, 224)

    # ── Tabular features ──
    num_raw     = np.array([[n, p, k, temp, hum, rain, ph, yld, fert]])
    num_sc      = scaler.transform(pd.DataFrame(num_raw, columns=num_cols))
    cat_enc     = np.array([[SEASON_MAP[season], IRRIG_MAP[irrig],
                              PREV_MAP[prev],     REGION_MAP[region]]])
    xgb_input = np.concatenate([num_sc, cat_enc], axis=1).astype(np.float32)

    xgb_probs = xgb_clf.predict_proba(xgb_input)                      # (1, 6)
    tab_raw   = np.concatenate([xgb_probs, xgb_input], axis=1).astype(np.float32)
    tab_t     = torch.tensor(tab_raw, dtype=torch.float32)             # (1, 19)

    # ── Inference — explicit eval() + no_grad every call ──────
    img_model.eval(); tab_proj.eval(); fusion.eval()
    with torch.no_grad():
        img_feat  = img_model(img_t, return_features=True)
        tab_feat  = tab_proj(tab_t)
        logits, _ = fusion(img_feat, tab_feat)

    # Clean prediction — raw softmax only, no blending or manipulation
    probs      = torch.softmax(logits, dim=-1)[0].cpu().numpy()        # (6,)
    pred_idx   = int(np.argmax(probs))
    soil_name  = class_names[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 2)
    all_probs  = {class_names[i]: round(float(probs[i]) * 100, 2)
                  for i in range(len(class_names))}

    debug = {
        "probs":        {class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))},
        "img_feat_std": round(img_feat.std().item(), 4),
    }

    soil_fert = SOIL_FERT_MAP.get(soil_name,
                {"fertilizer": "NPK 14:14:14", "npk": "N:P:K = 60:30:30 kg/ha"})

    crops_all = CROP_MAP.get(
        (soil_name, season),
        CROP_MAP.get((soil_name, "Kharif"), ["Wheat", "Rice", "Maize"]))

    crop_recs = []
    for i, crop in enumerate(crops_all[:3]):
        cf = CROP_FERT_MAP.get(crop, {"fertilizer": "NPK 14:14:14", "npk": "60:40:20 kg/ha"})
        crop_recs.append({"name": crop, "rank": i + 1, "stars": 5 - i,
                           "fertilizer": cf["fertilizer"], "npk": cf["npk"]})

    return soil_name, confidence, all_probs, soil_fert, crop_recs, debug


# ══════════════════════════════════════════════════════════════
# SOIL IMAGE VALIDATOR
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_validator():
    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)
    model.load_state_dict(
        torch.load("soil_validator.pt", map_location="cpu")
    )
    model.eval()
    return model

def is_soil_image(pil_img):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    validator = load_validator()
    img_t = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = validator(img_t)
        prob = torch.softmax(out, dim=-1)[0]
    soil_prob = prob[1].item()
    return soil_prob > 0.60


# ══════════════════════════════════════════════════════════════
# CLIMATE DATA FETCHER
# ══════════════════════════════════════════════════════════════

DISTRICT_COORDS = {
    ("Telangana", "Adilabad"): (19.6641, 78.5320),
    ("Telangana", "Bhadradri Kothagudem"): (17.5567, 80.6167),
    ("Telangana", "Hanamkonda"): (17.9689, 79.5941),
    ("Telangana", "Hyderabad"): (17.3850, 78.4867),
    ("Telangana", "Jagtial"): (18.7940, 78.9140),
    ("Telangana", "Jangaon"): (17.7244, 79.1523),
    ("Telangana", "Jayashankar Bhupalpally"): (18.4333, 79.9167),
    ("Telangana", "Jogulamba Gadwal"): (16.2340, 77.8020),
    ("Telangana", "Kamareddy"): (18.3240, 78.3410),
    ("Telangana", "Karimnagar"): (18.4386, 79.1288),
    ("Telangana", "Khammam"): (17.2473, 80.1514),
    ("Telangana", "Komaram Bheem"): (19.2167, 79.4167),
    ("Telangana", "Mahabubabad"): (17.6010, 80.0010),
    ("Telangana", "Mahabubnagar"): (16.7488, 77.9827),
    ("Telangana", "Mancherial"): (18.8710, 79.4600),
    ("Telangana", "Medak"): (18.0440, 78.2620),
    ("Telangana", "Medchal Malkajgiri"): (17.5333, 78.5333),
    ("Telangana", "Mulugu"): (18.1833, 80.0333),
    ("Telangana", "Nagarkurnool"): (16.4833, 78.3167),
    ("Telangana", "Nalgonda"): (17.0575, 79.2671),
    ("Telangana", "Narayanpet"): (16.7417, 77.4942),
    ("Telangana", "Nirmal"): (19.0960, 78.3440),
    ("Telangana", "Nizamabad"): (18.6725, 78.0941),
    ("Telangana", "Peddapalli"): (18.6140, 79.3830),
    ("Telangana", "Rajanna Sircilla"): (18.3873, 78.8322),
    ("Telangana", "Rangareddy"): (17.2000, 78.4000),
    ("Telangana", "Sangareddy"): (17.6274, 78.0878),
    ("Telangana", "Siddipet"): (18.1018, 78.8520),
    ("Telangana", "Suryapet"): (17.1403, 79.6210),
    ("Telangana", "Vikarabad"): (17.3380, 77.9040),
    ("Telangana", "Wanaparthy"): (16.3620, 78.0610),
    ("Telangana", "Warangal"): (17.9784, 79.5941),
    ("Telangana", "Yadadri Bhuvanagiri"): (17.5833, 79.1667),
    ("Andhra Pradesh", "Alluri Sitharama Raju"): (17.9333, 82.5667),
    ("Andhra Pradesh", "Anakapalli"): (17.6910, 83.0050),
    ("Andhra Pradesh", "Anantapur"): (14.6819, 77.6006),
    ("Andhra Pradesh", "Annamayya"): (13.8500, 79.0167),
    ("Andhra Pradesh", "Bapatla"): (15.9043, 80.4670),
    ("Andhra Pradesh", "Chittoor"): (13.2172, 79.1003),
    ("Andhra Pradesh", "East Godavari"): (17.0005, 82.2400),
    ("Andhra Pradesh", "Eluru"): (16.7107, 81.0952),
    ("Andhra Pradesh", "Guntur"): (16.3067, 80.4365),
    ("Andhra Pradesh", "Kakinada"): (16.9891, 82.2475),
    ("Andhra Pradesh", "Konaseema"): (16.8167, 82.2333),
    ("Andhra Pradesh", "Krishna"): (16.6167, 80.8000),
    ("Andhra Pradesh", "Kurnool"): (15.8281, 78.0373),
    ("Andhra Pradesh", "Nandyal"): (15.4786, 78.4839),
    ("Andhra Pradesh", "Nellore"): (14.4426, 79.9865),
    ("Andhra Pradesh", "NTR"): (16.5193, 80.6305),
    ("Andhra Pradesh", "Palnadu"): (16.2000, 79.6333),
    ("Andhra Pradesh", "Parvathipuram Manyam"): (18.7833, 83.4333),
    ("Andhra Pradesh", "Prakasam"): (15.3490, 79.5747),
    ("Andhra Pradesh", "Sri Balaji"): (13.6288, 79.4192),
    ("Andhra Pradesh", "Sri Sathya Sai"): (14.1667, 77.7833),
    ("Andhra Pradesh", "Srikakulam"): (18.2949, 83.8938),
    ("Andhra Pradesh", "Visakhapatnam"): (17.6868, 83.2185),
    ("Andhra Pradesh", "Vizianagaram"): (18.1066, 83.3956),
    ("Andhra Pradesh", "West Godavari"): (16.9167, 81.3333),
    ("Andhra Pradesh", "YSR Kadapa"): (14.4673, 78.8242),
    ("Karnataka", "Bagalkot"): (16.1800, 75.6960),
    ("Karnataka", "Ballari"): (15.1394, 76.9214),
    ("Karnataka", "Belagavi"): (15.8497, 74.4977),
    ("Karnataka", "Bengaluru Rural"): (13.1986, 77.7066),
    ("Karnataka", "Bengaluru Urban"): (12.9716, 77.5946),
    ("Karnataka", "Bidar"): (17.9104, 77.5199),
    ("Karnataka", "Chamarajanagar"): (11.9271, 76.9432),
    ("Karnataka", "Chikkaballapur"): (13.4355, 77.7315),
    ("Karnataka", "Chikkamagaluru"): (13.3153, 75.7754),
    ("Karnataka", "Chitradurga"): (14.2294, 76.3983),
    ("Karnataka", "Dakshina Kannada"): (12.8438, 75.2479),
    ("Karnataka", "Davanagere"): (14.4644, 75.9218),
    ("Karnataka", "Dharwad"): (15.4589, 75.0078),
    ("Karnataka", "Gadag"): (15.4167, 75.6167),
    ("Karnataka", "Hassan"): (13.0033, 76.1004),
    ("Karnataka", "Haveri"): (14.7939, 75.3996),
    ("Karnataka", "Kalaburagi"): (17.3297, 76.8343),
    ("Karnataka", "Kodagu"): (12.3375, 75.8069),
    ("Karnataka", "Kolar"): (13.1357, 78.1294),
    ("Karnataka", "Koppal"): (15.3500, 76.1500),
    ("Karnataka", "Mandya"): (12.5218, 76.8951),
    ("Karnataka", "Mysuru"): (12.2958, 76.6394),
    ("Karnataka", "Raichur"): (16.2120, 77.3439),
    ("Karnataka", "Ramanagara"): (12.7157, 77.2819),
    ("Karnataka", "Shivamogga"): (13.9299, 75.5681),
    ("Karnataka", "Tumakuru"): (13.3379, 77.1173),
    ("Karnataka", "Udupi"): (13.3409, 74.7421),
    ("Karnataka", "Uttara Kannada"): (14.7860, 74.6680),
    ("Karnataka", "Vijayapura"): (16.8302, 75.7100),
    ("Karnataka", "Yadgir"): (16.7710, 77.1380),
    ("Maharashtra", "Ahmednagar"): (19.0952, 74.7496),
    ("Maharashtra", "Akola"): (20.7002, 77.0082),
    ("Maharashtra", "Amravati"): (20.9374, 77.7796),
    ("Maharashtra", "Aurangabad"): (19.8762, 75.3433),
    ("Maharashtra", "Beed"): (18.9890, 75.7560),
    ("Maharashtra", "Bhandara"): (21.1667, 79.6500),
    ("Maharashtra", "Buldhana"): (20.5292, 76.1842),
    ("Maharashtra", "Chandrapur"): (19.9615, 79.2961),
    ("Maharashtra", "Dhule"): (20.9042, 74.7749),
    ("Maharashtra", "Gadchiroli"): (20.1809, 80.0000),
    ("Maharashtra", "Gondia"): (21.4624, 80.1947),
    ("Maharashtra", "Hingoli"): (19.7160, 77.1490),
    ("Maharashtra", "Jalgaon"): (21.0077, 75.5626),
    ("Maharashtra", "Jalna"): (19.8347, 75.8816),
    ("Maharashtra", "Kolhapur"): (16.7050, 74.2433),
    ("Maharashtra", "Latur"): (18.4088, 76.5604),
    ("Maharashtra", "Mumbai City"): (18.9388, 72.8354),
    ("Maharashtra", "Mumbai Suburban"): (19.0760, 72.8777),
    ("Maharashtra", "Nagpur"): (21.1458, 79.0882),
    ("Maharashtra", "Nanded"): (19.1383, 77.3210),
    ("Maharashtra", "Nandurbar"): (21.3667, 74.2333),
    ("Maharashtra", "Nashik"): (19.9975, 73.7898),
    ("Maharashtra", "Osmanabad"): (18.1860, 76.0410),
    ("Maharashtra", "Palghar"): (19.6967, 72.7659),
    ("Maharashtra", "Parbhani"): (19.2704, 76.7740),
    ("Maharashtra", "Pune"): (18.5204, 73.8567),
    ("Maharashtra", "Raigad"): (18.5158, 73.1298),
    ("Maharashtra", "Ratnagiri"): (16.9902, 73.3120),
    ("Maharashtra", "Sangli"): (16.8524, 74.5815),
    ("Maharashtra", "Satara"): (17.6805, 74.0183),
    ("Maharashtra", "Sindhudurg"): (16.0494, 73.5283),
    ("Maharashtra", "Solapur"): (17.6599, 75.9064),
    ("Maharashtra", "Thane"): (19.2183, 72.9781),
    ("Maharashtra", "Wardha"): (20.7453, 78.6022),
    ("Maharashtra", "Washim"): (20.1120, 77.1340),
    ("Maharashtra", "Yavatmal"): (20.3888, 78.1204),
    ("Punjab", "Amritsar"): (31.6340, 74.8723),
    ("Punjab", "Barnala"): (30.3782, 75.5492),
    ("Punjab", "Bathinda"): (30.2110, 74.9455),
    ("Punjab", "Faridkot"): (30.6717, 74.7553),
    ("Punjab", "Fatehgarh Sahib"): (30.6480, 76.3906),
    ("Punjab", "Fazilka"): (30.4019, 74.0257),
    ("Punjab", "Ferozepur"): (30.9236, 74.6227),
    ("Punjab", "Gurdaspur"): (32.0399, 75.4060),
    ("Punjab", "Hoshiarpur"): (31.5143, 75.9119),
    ("Punjab", "Jalandhar"): (31.3260, 75.5762),
    ("Punjab", "Kapurthala"): (31.3800, 75.3800),
    ("Punjab", "Ludhiana"): (30.9010, 75.8573),
    ("Punjab", "Mansa"): (29.9918, 75.3980),
    ("Punjab", "Moga"): (30.8170, 75.1730),
    ("Punjab", "Mohali"): (30.7046, 76.7179),
    ("Punjab", "Muktsar"): (30.4740, 74.5160),
    ("Punjab", "Pathankot"): (32.2743, 75.6522),
    ("Punjab", "Patiala"): (30.3398, 76.3869),
    ("Punjab", "Rupnagar"): (30.9644, 76.5254),
    ("Punjab", "Sangrur"): (30.2457, 75.8425),
    ("Punjab", "Shaheed Bhagat Singh Nagar"): (31.1270, 76.3870),
    ("Punjab", "Tarn Taran"): (31.4520, 74.9280),
    ("Haryana", "Ambala"): (30.3752, 76.7821),
    ("Haryana", "Bhiwani"): (28.7975, 76.1322),
    ("Haryana", "Charkhi Dadri"): (28.5921, 76.2700),
    ("Haryana", "Faridabad"): (28.4089, 77.3178),
    ("Haryana", "Fatehabad"): (29.5136, 75.4551),
    ("Haryana", "Gurugram"): (28.4595, 77.0266),
    ("Haryana", "Hisar"): (29.1492, 75.7217),
    ("Haryana", "Jhajjar"): (28.6080, 76.6572),
    ("Haryana", "Jind"): (29.3162, 76.3163),
    ("Haryana", "Kaithal"): (29.8014, 76.3998),
    ("Haryana", "Karnal"): (29.6857, 76.9905),
    ("Haryana", "Kurukshetra"): (29.9695, 76.8783),
    ("Haryana", "Mahendragarh"): (28.2785, 76.1458),
    ("Haryana", "Nuh"): (28.1075, 77.0006),
    ("Haryana", "Palwal"): (28.1487, 77.3270),
    ("Haryana", "Panchkula"): (30.6942, 76.8606),
    ("Haryana", "Panipat"): (29.3909, 76.9635),
    ("Haryana", "Rewari"): (28.1972, 76.6172),
    ("Haryana", "Rohtak"): (28.8955, 76.6066),
    ("Haryana", "Sirsa"): (29.5330, 75.0280),
    ("Haryana", "Sonipat"): (28.9931, 77.0151),
    ("Haryana", "Yamunanagar"): (30.1290, 77.2674),
    ("Gujarat", "Ahmedabad"): (23.0225, 72.5714),
    ("Gujarat", "Amreli"): (21.6032, 71.2215),
    ("Gujarat", "Anand"): (22.5645, 72.9289),
    ("Gujarat", "Aravalli"): (23.6993, 73.1209),
    ("Gujarat", "Banaskantha"): (24.1740, 72.4370),
    ("Gujarat", "Bharuch"): (21.7051, 72.9959),
    ("Gujarat", "Bhavnagar"): (21.7645, 72.1519),
    ("Gujarat", "Botad"): (22.1690, 71.6680),
    ("Gujarat", "Chhota Udaipur"): (22.3063, 74.0146),
    ("Gujarat", "Dahod"): (22.8340, 74.2560),
    ("Gujarat", "Dang"): (20.7500, 73.6700),
    ("Gujarat", "Devbhoomi Dwarka"): (22.2394, 68.9678),
    ("Gujarat", "Gandhinagar"): (23.2156, 72.6369),
    ("Gujarat", "Gir Somnath"): (20.9060, 70.3700),
    ("Gujarat", "Jamnagar"): (22.4707, 70.0577),
    ("Gujarat", "Junagadh"): (21.5222, 70.4579),
    ("Gujarat", "Kheda"): (22.7500, 72.6800),
    ("Gujarat", "Kutch"): (23.7337, 69.8597),
    ("Gujarat", "Mahisagar"): (23.1000, 73.5900),
    ("Gujarat", "Mehsana"): (23.5879, 72.3693),
    ("Gujarat", "Morbi"): (22.8173, 70.8372),
    ("Gujarat", "Narmada"): (21.8716, 73.4979),
    ("Gujarat", "Navsari"): (20.9467, 72.9520),
    ("Gujarat", "Panchmahal"): (22.7500, 73.5800),
    ("Gujarat", "Patan"): (23.8493, 72.1266),
    ("Gujarat", "Porbandar"): (21.6425, 69.6293),
    ("Gujarat", "Rajkot"): (22.3039, 70.8022),
    ("Gujarat", "Sabarkantha"): (23.3800, 73.0100),
    ("Gujarat", "Surat"): (21.1702, 72.8311),
    ("Gujarat", "Surendranagar"): (22.7270, 71.6472),
    ("Gujarat", "Tapi"): (21.1200, 73.4100),
    ("Gujarat", "Vadodara"): (22.3072, 73.1812),
    ("Gujarat", "Valsad"): (20.5992, 72.9342),
    ("Rajasthan", "Ajmer"): (26.4499, 74.6399),
    ("Rajasthan", "Alwar"): (27.5530, 76.6346),
    ("Rajasthan", "Banswara"): (23.5500, 74.4400),
    ("Rajasthan", "Baran"): (25.1000, 76.5200),
    ("Rajasthan", "Barmer"): (25.7500, 71.3800),
    ("Rajasthan", "Bharatpur"): (27.2152, 77.4938),
    ("Rajasthan", "Bhilwara"): (25.3500, 74.6400),
    ("Rajasthan", "Bikaner"): (28.0229, 73.3119),
    ("Rajasthan", "Bundi"): (25.4395, 75.6390),
    ("Rajasthan", "Chittorgarh"): (24.8887, 74.6269),
    ("Rajasthan", "Churu"): (28.2960, 74.9640),
    ("Rajasthan", "Dausa"): (26.8934, 76.3397),
    ("Rajasthan", "Dholpur"): (26.7010, 77.8940),
    ("Rajasthan", "Dungarpur"): (23.8400, 73.7200),
    ("Rajasthan", "Hanumangarh"): (29.5826, 74.3292),
    ("Rajasthan", "Jaipur"): (26.9124, 75.7873),
    ("Rajasthan", "Jaisalmer"): (26.9157, 70.9083),
    ("Rajasthan", "Jalore"): (25.3500, 72.6200),
    ("Rajasthan", "Jhalawar"): (24.5975, 76.1650),
    ("Rajasthan", "Jhunjhunu"): (28.1290, 75.3990),
    ("Rajasthan", "Jodhpur"): (26.2389, 73.0243),
    ("Rajasthan", "Karauli"): (26.5000, 77.0200),
    ("Rajasthan", "Kota"): (25.2138, 75.8648),
    ("Rajasthan", "Nagaur"): (27.2025, 73.7285),
    ("Rajasthan", "Pali"): (25.7730, 73.3234),
    ("Rajasthan", "Pratapgarh"): (24.0330, 74.7780),
    ("Rajasthan", "Rajsamand"): (25.0700, 73.8800),
    ("Rajasthan", "Sawai Madhopur"): (25.9964, 76.3545),
    ("Rajasthan", "Sikar"): (27.6094, 75.1399),
    ("Rajasthan", "Sirohi"): (24.8860, 72.8620),
    ("Rajasthan", "Sri Ganganagar"): (29.9166, 73.8833),
    ("Rajasthan", "Tonk"): (26.1630, 75.7880),
    ("Rajasthan", "Udaipur"): (24.5854, 73.7125),
    ("Kerala", "Alappuzha"): (9.4981, 76.3388),
    ("Kerala", "Ernakulam"): (9.9816, 76.2999),
    ("Kerala", "Idukki"): (9.9189, 77.1025),
    ("Kerala", "Kannur"): (11.8745, 75.3704),
    ("Kerala", "Kasaragod"): (12.4996, 74.9869),
    ("Kerala", "Kollam"): (8.8932, 76.6141),
    ("Kerala", "Kottayam"): (9.5916, 76.5222),
    ("Kerala", "Kozhikode"): (11.2588, 75.7804),
    ("Kerala", "Malappuram"): (11.0510, 76.0711),
    ("Kerala", "Palakkad"): (10.7867, 76.6548),
    ("Kerala", "Pathanamthitta"): (9.2648, 76.7870),
    ("Kerala", "Thiruvananthapuram"): (8.5241, 76.9366),
    ("Kerala", "Thrissur"): (10.5276, 76.2144),
    ("Kerala", "Wayanad"): (11.6854, 76.1320),
    ("Tamil Nadu", "Ariyalur"): (11.1437, 79.0747),
    ("Tamil Nadu", "Chengalpattu"): (12.6921, 79.9757),
    ("Tamil Nadu", "Chennai"): (13.0827, 80.2707),
    ("Tamil Nadu", "Coimbatore"): (11.0168, 76.9558),
    ("Tamil Nadu", "Cuddalore"): (11.7480, 79.7714),
    ("Tamil Nadu", "Dharmapuri"): (12.1211, 78.1582),
    ("Tamil Nadu", "Dindigul"): (10.3624, 77.9695),
    ("Tamil Nadu", "Erode"): (11.3410, 77.7172),
    ("Tamil Nadu", "Kallakurichi"): (11.7380, 78.9590),
    ("Tamil Nadu", "Kancheepuram"): (12.8185, 79.7018),
    ("Tamil Nadu", "Kanyakumari"): (8.0883, 77.5385),
    ("Tamil Nadu", "Karur"): (10.9601, 78.0766),
    ("Tamil Nadu", "Krishnagiri"): (12.5266, 78.2138),
    ("Tamil Nadu", "Madurai"): (9.9252, 78.1198),
    ("Tamil Nadu", "Mayiladuthurai"): (11.1035, 79.6508),
    ("Tamil Nadu", "Nagapattinam"): (10.7672, 79.8449),
    ("Tamil Nadu", "Namakkal"): (11.2342, 78.1674),
    ("Tamil Nadu", "Nilgiris"): (11.4916, 76.7337),
    ("Tamil Nadu", "Perambalur"): (11.2340, 78.8800),
    ("Tamil Nadu", "Pudukkottai"): (10.3797, 78.8200),
    ("Tamil Nadu", "Ramanathapuram"): (9.3639, 78.8395),
    ("Tamil Nadu", "Ranipet"): (12.9310, 79.3330),
    ("Tamil Nadu", "Salem"): (11.6643, 78.1460),
    ("Tamil Nadu", "Sivaganga"): (9.8479, 78.4800),
    ("Tamil Nadu", "Tenkasi"): (8.9590, 77.3150),
    ("Tamil Nadu", "Thanjavur"): (10.7870, 79.1378),
    ("Tamil Nadu", "Theni"): (10.0104, 77.4770),
    ("Tamil Nadu", "Thoothukudi"): (8.7642, 78.1348),
    ("Tamil Nadu", "Tiruchirappalli"): (10.7905, 78.7047),
    ("Tamil Nadu", "Tirunelveli"): (8.7139, 77.7567),
    ("Tamil Nadu", "Tirupathur"): (12.4960, 78.5720),
    ("Tamil Nadu", "Tiruppur"): (11.1085, 77.3411),
    ("Tamil Nadu", "Tiruvallur"): (13.1231, 79.9099),
    ("Tamil Nadu", "Tiruvannamalai"): (12.2253, 79.0747),
    ("Tamil Nadu", "Tiruvarur"): (10.7725, 79.6367),
    ("Tamil Nadu", "Vellore"): (12.9165, 79.1325),
    ("Tamil Nadu", "Viluppuram"): (11.9401, 79.4861),
    ("Tamil Nadu", "Virudhunagar"): (9.5851, 77.9624),
    ("Bihar", "Araria"): (26.1500, 87.4700), ("Bihar", "Arwal"): (25.2500, 84.6800),
    ("Bihar", "Aurangabad"): (24.7500, 84.3700), ("Bihar", "Banka"): (24.8800, 86.9200),
    ("Bihar", "Begusarai"): (25.4180, 86.1290), ("Bihar", "Bhagalpur"): (25.2500, 86.9833),
    ("Bihar", "Bhojpur"): (25.5600, 84.4500), ("Bihar", "Buxar"): (25.5600, 83.9800),
    ("Bihar", "Darbhanga"): (26.1542, 85.8918), ("Bihar", "East Champaran"): (26.6500, 84.9200),
    ("Bihar", "Gaya"): (24.7914, 85.0002), ("Bihar", "Gopalganj"): (26.4671, 84.4376),
    ("Bihar", "Jamui"): (24.9200, 86.2200), ("Bihar", "Jehanabad"): (25.2100, 84.9900),
    ("Bihar", "Kaimur"): (25.0500, 83.6000), ("Bihar", "Katihar"): (25.5379, 87.5765),
    ("Bihar", "Khagaria"): (25.5000, 86.4600), ("Bihar", "Kishanganj"): (26.1000, 87.9500),
    ("Bihar", "Lakhisarai"): (25.1500, 86.1000), ("Bihar", "Madhepura"): (25.9200, 86.7900),
    ("Bihar", "Madhubani"): (26.3500, 86.0800), ("Bihar", "Munger"): (25.3700, 86.4700),
    ("Bihar", "Muzaffarpur"): (26.1225, 85.3906), ("Bihar", "Nalanda"): (25.1373, 85.4440),
    ("Bihar", "Nawada"): (24.8800, 85.5400), ("Bihar", "Patna"): (25.5941, 85.1376),
    ("Bihar", "Purnia"): (25.7771, 87.4753), ("Bihar", "Rohtas"): (24.9900, 84.0300),
    ("Bihar", "Saharsa"): (25.8800, 86.6000), ("Bihar", "Samastipur"): (25.8637, 85.7811),
    ("Bihar", "Saran"): (25.9200, 84.8500), ("Bihar", "Sheikhpura"): (25.1400, 85.8500),
    ("Bihar", "Sheohar"): (26.5200, 85.2900), ("Bihar", "Sitamarhi"): (26.5900, 85.4900),
    ("Bihar", "Siwan"): (26.2200, 84.3600), ("Bihar", "Supaul"): (26.1200, 86.6000),
    ("Bihar", "Vaishali"): (25.6900, 85.2000), ("Bihar", "West Champaran"): (27.1500, 84.3500),
    ("West Bengal", "Alipurduar"): (26.4900, 89.5300), ("West Bengal", "Bankura"): (23.2500, 87.0667),
    ("West Bengal", "Birbhum"): (23.9000, 87.5333), ("West Bengal", "Cooch Behar"): (26.3200, 89.4500),
    ("West Bengal", "Dakshin Dinajpur"): (25.3200, 88.7500), ("West Bengal", "Darjeeling"): (27.0360, 88.2627),
    ("West Bengal", "Hooghly"): (22.9000, 88.4000), ("West Bengal", "Howrah"): (22.5958, 88.2636),
    ("West Bengal", "Jalpaiguri"): (26.5163, 88.7274), ("West Bengal", "Jhargram"): (22.4500, 86.9900),
    ("West Bengal", "Kalimpong"): (27.0600, 88.4700), ("West Bengal", "Kolkata"): (22.5726, 88.3639),
    ("West Bengal", "Malda"): (25.0108, 88.1415), ("West Bengal", "Murshidabad"): (24.1800, 88.2700),
    ("West Bengal", "Nadia"): (23.4000, 88.5500), ("West Bengal", "North 24 Parganas"): (22.8700, 88.5800),
    ("West Bengal", "Paschim Bardhaman"): (23.2333, 87.0833), ("West Bengal", "Paschim Medinipur"): (22.4167, 87.3167),
    ("West Bengal", "Purba Bardhaman"): (23.2500, 87.8500), ("West Bengal", "Purba Medinipur"): (22.0833, 87.7500),
    ("West Bengal", "Purulia"): (23.3333, 86.3667), ("West Bengal", "South 24 Parganas"): (22.1500, 88.6800),
    ("West Bengal", "Uttar Dinajpur"): (26.1200, 88.1200),
    ("Madhya Pradesh", "Bhopal"): (23.2599, 77.4126), ("Madhya Pradesh", "Indore"): (22.7196, 75.8577),
    ("Madhya Pradesh", "Jabalpur"): (23.1815, 79.9864), ("Madhya Pradesh", "Gwalior"): (26.2183, 78.1828),
    ("Madhya Pradesh", "Ujjain"): (23.1765, 75.7885), ("Madhya Pradesh", "Sagar"): (23.8388, 78.7378),
    ("Madhya Pradesh", "Rewa"): (24.5362, 81.2997), ("Madhya Pradesh", "Satna"): (24.5800, 80.8300),
    ("Madhya Pradesh", "Chhindwara"): (22.0574, 78.9382), ("Madhya Pradesh", "Vidisha"): (23.5251, 77.8093),
    ("Madhya Pradesh", "Hoshangabad"): (22.7500, 77.7200), ("Madhya Pradesh", "Damoh"): (23.8310, 79.4420),
    ("Madhya Pradesh", "Narsinghpur"): (22.9477, 79.1940), ("Madhya Pradesh", "Khandwa"): (21.8274, 76.3520),
    ("Madhya Pradesh", "Khargone"): (21.8230, 75.6110), ("Madhya Pradesh", "Ratlam"): (23.3313, 75.0367),
    ("Madhya Pradesh", "Mandsaur"): (24.0700, 75.0700), ("Madhya Pradesh", "Neemuch"): (24.4700, 74.8700),
    ("Madhya Pradesh", "Dhar"): (22.5982, 75.2996), ("Madhya Pradesh", "Jhabua"): (22.7700, 74.5900),
    ("Madhya Pradesh", "Alirajpur"): (22.3040, 74.3570), ("Madhya Pradesh", "Barwani"): (22.0320, 74.9020),
    ("Madhya Pradesh", "Betul"): (21.9061, 77.8978), ("Madhya Pradesh", "Bhind"): (26.5647, 78.7877),
    ("Madhya Pradesh", "Chhatarpur"): (24.9100, 79.5900), ("Madhya Pradesh", "Datia"): (25.6700, 78.4600),
    ("Madhya Pradesh", "Dewas"): (22.9676, 76.0534), ("Madhya Pradesh", "Dindori"): (22.9400, 81.0800),
    ("Madhya Pradesh", "Guna"): (24.6479, 77.3148), ("Madhya Pradesh", "Harda"): (22.3417, 77.0900),
    ("Madhya Pradesh", "Katni"): (23.8331, 80.3964), ("Madhya Pradesh", "Mandla"): (22.5981, 80.3764),
    ("Madhya Pradesh", "Morena"): (26.4960, 77.9980), ("Madhya Pradesh", "Panna"): (24.7175, 80.1853),
    ("Madhya Pradesh", "Raisen"): (23.3300, 77.7900), ("Madhya Pradesh", "Rajgarh"): (24.0219, 76.7220),
    ("Madhya Pradesh", "Seoni"): (22.0854, 79.5389), ("Madhya Pradesh", "Shahdol"): (23.2977, 81.3567),
    ("Madhya Pradesh", "Shajapur"): (23.4280, 76.2770), ("Madhya Pradesh", "Sheopur"): (25.6700, 76.7100),
    ("Madhya Pradesh", "Shivpuri"): (25.4231, 77.6590), ("Madhya Pradesh", "Sidhi"): (24.4161, 81.8769),
    ("Madhya Pradesh", "Singrauli"): (24.1997, 82.6746), ("Madhya Pradesh", "Tikamgarh"): (24.7400, 78.8300),
    ("Madhya Pradesh", "Umaria"): (23.5200, 80.8400), ("Madhya Pradesh", "Agar Malwa"): (23.7100, 76.0200),
    ("Madhya Pradesh", "Anuppur"): (23.1000, 81.6900), ("Madhya Pradesh", "Ashoknagar"): (24.5700, 77.7300),
    ("Madhya Pradesh", "Balaghat"): (21.8138, 80.1867), ("Madhya Pradesh", "Burhanpur"): (21.3100, 76.2300),
    ("Madhya Pradesh", "Niwari"): (25.1100, 78.9900),
    ("Uttar Pradesh", "Lucknow"): (26.8467, 80.9462), ("Uttar Pradesh", "Agra"): (27.1767, 78.0081),
    ("Uttar Pradesh", "Kanpur Nagar"): (26.4499, 80.3319), ("Uttar Pradesh", "Varanasi"): (25.3176, 82.9739),
    ("Uttar Pradesh", "Prayagraj"): (25.4358, 81.8463), ("Uttar Pradesh", "Meerut"): (28.9845, 77.7064),
    ("Uttar Pradesh", "Ghaziabad"): (28.6692, 77.4538), ("Uttar Pradesh", "Mathura"): (27.4924, 77.6737),
    ("Uttar Pradesh", "Aligarh"): (27.8974, 78.0880), ("Uttar Pradesh", "Bareilly"): (28.3670, 79.4304),
    ("Uttar Pradesh", "Gorakhpur"): (26.7606, 83.3732), ("Uttar Pradesh", "Moradabad"): (28.8386, 78.7733),
    ("Uttar Pradesh", "Saharanpur"): (29.9640, 77.5460), ("Uttar Pradesh", "Muzaffarnagar"): (29.4727, 77.7085),
    ("Uttar Pradesh", "Firozabad"): (27.1592, 78.3957), ("Uttar Pradesh", "Jhansi"): (25.4484, 78.5685),
    ("Uttar Pradesh", "Ayodhya"): (26.7922, 82.1998), ("Uttar Pradesh", "Azamgarh"): (26.0686, 83.1838),
    ("Uttar Pradesh", "Lakhimpur Kheri"): (27.9487, 80.7821), ("Uttar Pradesh", "Sultanpur"): (26.2648, 82.0727),
    ("Uttar Pradesh", "Bulandshahr"): (28.4069, 77.8497), ("Uttar Pradesh", "Sitapur"): (27.5680, 80.6830),
    ("Uttar Pradesh", "Bijnor"): (29.3719, 78.1350), ("Uttar Pradesh", "Shahjahanpur"): (27.8830, 79.9050),
    ("Uttar Pradesh", "Ghazipur"): (25.5782, 83.5786), ("Uttar Pradesh", "Ballia"): (25.7592, 84.1493),
    ("Uttar Pradesh", "Jaunpur"): (25.7458, 82.6838), ("Uttar Pradesh", "Mirzapur"): (25.1460, 82.5690),
    ("Uttar Pradesh", "Sonbhadra"): (24.6852, 83.0694), ("Uttar Pradesh", "Basti"): (26.8013, 82.7227),
    ("Uttar Pradesh", "Deoria"): (26.5014, 83.7806), ("Uttar Pradesh", "Kushinagar"): (26.7400, 83.8890),
    ("Uttar Pradesh", "Maharajganj"): (27.1300, 83.5700), ("Uttar Pradesh", "Siddharthnagar"): (27.2900, 83.0700),
    ("Uttar Pradesh", "Sant Kabir Nagar"): (26.7900, 83.0600), ("Uttar Pradesh", "Ambedkar Nagar"): (26.4600, 82.5900),
    ("Uttar Pradesh", "Amethi"): (26.1500, 81.9300), ("Uttar Pradesh", "Rae Bareli"): (26.2309, 81.2317),
    ("Uttar Pradesh", "Unnao"): (26.5460, 80.4910), ("Uttar Pradesh", "Hardoi"): (27.3952, 80.1317),
    ("Uttar Pradesh", "Farrukhabad"): (27.3880, 79.5800), ("Uttar Pradesh", "Kannauj"): (27.0571, 79.9178),
    ("Uttar Pradesh", "Etawah"): (26.7857, 79.0246), ("Uttar Pradesh", "Auraiya"): (26.4700, 79.5100),
    ("Uttar Pradesh", "Mainpuri"): (27.2352, 79.0241), ("Uttar Pradesh", "Hathras"): (27.5915, 78.0500),
    ("Uttar Pradesh", "Kasganj"): (27.8100, 78.6400), ("Uttar Pradesh", "Etah"): (27.5593, 78.6667),
    ("Uttar Pradesh", "Budaun"): (28.0400, 79.1200), ("Uttar Pradesh", "Rampur"): (28.7952, 79.0155),
    ("Uttar Pradesh", "Amroha"): (28.9040, 78.4680), ("Uttar Pradesh", "Sambhal"): (28.5900, 78.5700),
    ("Uttar Pradesh", "Pilibhit"): (28.6400, 79.8000), ("Uttar Pradesh", "Baghpat"): (28.9400, 77.2200),
    ("Uttar Pradesh", "Hapur"): (28.7300, 77.7800), ("Uttar Pradesh", "Gautam Buddha Nagar"): (28.5355, 77.3910),
    ("Uttar Pradesh", "Mau"): (25.9418, 83.5610), ("Uttar Pradesh", "Chandauli"): (25.2700, 83.2700),
    ("Uttar Pradesh", "Bhadohi"): (25.3900, 82.5700), ("Uttar Pradesh", "Kaushambi"): (25.5300, 81.3800),
    ("Uttar Pradesh", "Fatehpur"): (25.9300, 80.8100), ("Uttar Pradesh", "Pratapgarh"): (25.9000, 81.9900),
    ("Uttar Pradesh", "Banda"): (25.4756, 80.3342), ("Uttar Pradesh", "Chitrakoot"): (25.2000, 80.8800),
    ("Uttar Pradesh", "Hamirpur"): (25.9500, 80.1500), ("Uttar Pradesh", "Mahoba"): (25.2900, 79.8700),
    ("Uttar Pradesh", "Jalaun"): (26.1400, 79.3300), ("Uttar Pradesh", "Lalitpur"): (24.6878, 78.4159),
    ("Uttar Pradesh", "Shravasti"): (27.5100, 81.8400), ("Uttar Pradesh", "Balrampur"): (27.4300, 82.1800),
    ("Uttar Pradesh", "Bahraich"): (27.5742, 81.5961), ("Uttar Pradesh", "Gonda"): (27.1300, 81.9700),
    ("Uttar Pradesh", "Barabanki"): (26.9200, 81.1800), ("Uttar Pradesh", "Shamli"): (29.4500, 77.3100),
    ("Odisha", "Angul"): (20.8400, 85.1000), ("Odisha", "Balangir"): (20.7000, 83.4900),
    ("Odisha", "Balasore"): (21.4942, 86.9336), ("Odisha", "Bargarh"): (21.3300, 83.6100),
    ("Odisha", "Bhadrak"): (21.0541, 86.5133), ("Odisha", "Boudh"): (20.8400, 84.3200),
    ("Odisha", "Cuttack"): (20.4625, 85.8828), ("Odisha", "Deogarh"): (21.5400, 84.7300),
    ("Odisha", "Dhenkanal"): (20.6600, 85.5900), ("Odisha", "Gajapati"): (19.0700, 84.1100),
    ("Odisha", "Ganjam"): (19.3900, 84.9800), ("Odisha", "Jagatsinghpur"): (20.2600, 86.1700),
    ("Odisha", "Jajpur"): (20.8500, 86.3400), ("Odisha", "Jharsuguda"): (21.8550, 84.0060),
    ("Odisha", "Kalahandi"): (19.9100, 83.1700), ("Odisha", "Kandhamal"): (20.1100, 84.2300),
    ("Odisha", "Kendrapara"): (20.5000, 86.4200), ("Odisha", "Kendujhar"): (21.6300, 85.5800),
    ("Odisha", "Khordha"): (20.1800, 85.6100), ("Odisha", "Koraput"): (18.8100, 82.7100),
    ("Odisha", "Malkangiri"): (18.3500, 81.9000), ("Odisha", "Mayurbhanj"): (21.9400, 86.2700),
    ("Odisha", "Nabarangpur"): (19.2300, 82.5500), ("Odisha", "Nayagarh"): (20.1300, 85.0900),
    ("Odisha", "Nuapada"): (20.8300, 82.5400), ("Odisha", "Puri"): (19.8135, 85.8312),
    ("Odisha", "Rayagada"): (19.1700, 83.4200), ("Odisha", "Sambalpur"): (21.4669, 83.9812),
    ("Odisha", "Subarnapur"): (20.8300, 83.9000), ("Odisha", "Sundargarh"): (22.1167, 84.0333),
    ("Jharkhand", "Bokaro"): (23.6693, 86.1511), ("Jharkhand", "Chatra"): (24.2100, 84.8800),
    ("Jharkhand", "Deoghar"): (24.4800, 86.7000), ("Jharkhand", "Dhanbad"): (23.7998, 86.4305),
    ("Jharkhand", "Dumka"): (24.2700, 87.2500), ("Jharkhand", "East Singhbhum"): (22.8046, 86.2029),
    ("Jharkhand", "Garhwa"): (24.1600, 83.8100), ("Jharkhand", "Giridih"): (24.1900, 86.3000),
    ("Jharkhand", "Godda"): (24.8300, 87.2100), ("Jharkhand", "Gumla"): (23.0500, 84.5400),
    ("Jharkhand", "Hazaribagh"): (23.9925, 85.3617), ("Jharkhand", "Jamtara"): (23.9600, 86.8000),
    ("Jharkhand", "Khunti"): (23.0700, 85.2800), ("Jharkhand", "Koderma"): (24.4600, 85.6000),
    ("Jharkhand", "Latehar"): (23.7400, 84.5000), ("Jharkhand", "Lohardaga"): (23.4400, 84.6800),
    ("Jharkhand", "Pakur"): (24.6400, 87.8400), ("Jharkhand", "Palamu"): (24.0300, 84.0700),
    ("Jharkhand", "Ramgarh"): (23.6400, 85.5100), ("Jharkhand", "Ranchi"): (23.3441, 85.3096),
    ("Jharkhand", "Sahebganj"): (25.2400, 87.6600), ("Jharkhand", "Seraikela Kharsawan"): (22.5800, 85.9900),
    ("Jharkhand", "Simdega"): (22.6100, 84.5200), ("Jharkhand", "West Singhbhum"): (22.1500, 85.6500),
    ("Chhattisgarh", "Raipur"): (21.2514, 81.6296), ("Chhattisgarh", "Bilaspur"): (22.0796, 82.1391),
    ("Chhattisgarh", "Durg"): (21.1900, 81.2800), ("Chhattisgarh", "Rajnandgaon"): (21.0970, 81.0290),
    ("Chhattisgarh", "Korba"): (22.3595, 82.7501), ("Chhattisgarh", "Bastar"): (19.1100, 81.9500),
    ("Chhattisgarh", "Raigarh"): (21.8974, 83.3950), ("Chhattisgarh", "Kanker"): (20.2700, 81.4900),
    ("Chhattisgarh", "Mahasamund"): (21.1100, 82.0900), ("Chhattisgarh", "Dhamtari"): (20.7100, 81.5500),
    ("Chhattisgarh", "Janjgir Champa"): (22.0100, 82.5700), ("Chhattisgarh", "Gariaband"): (20.6300, 82.0600),
    ("Chhattisgarh", "Balod"): (20.7300, 81.2100), ("Chhattisgarh", "Baloda Bazar"): (21.6600, 82.1600),
    ("Chhattisgarh", "Balrampur"): (23.1300, 83.6000), ("Chhattisgarh", "Bemetara"): (21.7100, 81.5300),
    ("Chhattisgarh", "Bijapur"): (18.8300, 80.2500), ("Chhattisgarh", "Dantewada"): (18.8900, 81.3500),
    ("Chhattisgarh", "Gaurela Pendra Marwahi"): (22.7500, 81.7900), ("Chhattisgarh", "Jashpur"): (22.8800, 84.1500),
    ("Chhattisgarh", "Kabirdham"): (22.0000, 81.2700), ("Chhattisgarh", "Khairagarh"): (21.4200, 80.9800),
    ("Chhattisgarh", "Kondagaon"): (19.5900, 81.6600), ("Chhattisgarh", "Koriya"): (23.2500, 82.7000),
    ("Chhattisgarh", "Manendragarh"): (23.2000, 82.2200), ("Chhattisgarh", "Mohla Manpur"): (20.8000, 80.7900),
    ("Chhattisgarh", "Mungeli"): (22.0600, 81.6900), ("Chhattisgarh", "Narayanpur"): (19.6900, 81.2400),
    ("Chhattisgarh", "Sakti"): (22.0300, 82.9700), ("Chhattisgarh", "Sarangarh Bilaigarh"): (21.5800, 83.0700),
    ("Chhattisgarh", "Sukma"): (18.3900, 81.6600), ("Chhattisgarh", "Surajpur"): (23.2200, 82.8600),
    ("Chhattisgarh", "Surguja"): (23.1200, 83.1900),
    ("Assam", "Kamrup Metropolitan"): (26.1445, 91.7362), ("Assam", "Kamrup"): (26.1000, 91.4000),
    ("Assam", "Dibrugarh"): (27.4728, 94.9120), ("Assam", "Jorhat"): (26.7509, 94.2037),
    ("Assam", "Nagaon"): (26.3472, 92.6839), ("Assam", "Sonitpur"): (26.6300, 92.8000),
    ("Assam", "Sivasagar"): (26.9800, 94.6300), ("Assam", "Tinsukia"): (27.4894, 95.3559),
    ("Assam", "Lakhimpur"): (27.2348, 94.1000), ("Assam", "Dhemaji"): (27.4800, 94.5600),
    ("Assam", "Barpeta"): (26.3200, 91.0000), ("Assam", "Nalbari"): (26.4400, 91.4300),
    ("Assam", "Baksa"): (26.6400, 91.2000), ("Assam", "Darrang"): (26.4500, 92.1700),
    ("Assam", "Udalguri"): (26.7500, 92.1000), ("Assam", "Dhubri"): (26.0200, 89.9900),
    ("Assam", "Goalpara"): (26.1700, 90.6200), ("Assam", "Bongaigaon"): (26.4800, 90.5600),
    ("Assam", "Chirang"): (26.5200, 90.4700), ("Assam", "Kokrajhar"): (26.4000, 90.2700),
    ("Assam", "Cachar"): (24.8333, 92.7789), ("Assam", "Hailakandi"): (24.6900, 92.5600),
    ("Assam", "Karimganj"): (24.8600, 92.3600), ("Assam", "Dima Hasao"): (25.5700, 93.0200),
    ("Assam", "Karbi Anglong"): (26.0000, 93.5000), ("Assam", "West Karbi Anglong"): (25.9000, 93.0000),
    ("Assam", "Golaghat"): (26.5200, 93.9700), ("Assam", "Majuli"): (26.9500, 94.1600),
    ("Assam", "Biswanath"): (26.7300, 93.1500), ("Assam", "Charaideo"): (27.0100, 94.8100),
    ("Assam", "Hojai"): (26.0000, 92.8500), ("Assam", "Morigaon"): (26.2500, 92.3300),
    ("Assam", "South Salmara"): (25.9400, 89.8700), ("Assam", "Bajali"): (26.4600, 91.1500),
    ("Himachal Pradesh", "Shimla"): (31.1048, 77.1734), ("Himachal Pradesh", "Kangra"): (32.0998, 76.2691),
    ("Himachal Pradesh", "Mandi"): (31.7088, 76.9318), ("Himachal Pradesh", "Kullu"): (31.9579, 77.1095),
    ("Himachal Pradesh", "Solan"): (30.9045, 77.0967), ("Himachal Pradesh", "Una"): (31.4685, 76.2709),
    ("Himachal Pradesh", "Hamirpur"): (31.6862, 76.5215), ("Himachal Pradesh", "Bilaspur"): (31.3314, 76.7605),
    ("Himachal Pradesh", "Chamba"): (32.5534, 76.1258), ("Himachal Pradesh", "Sirmaur"): (30.5600, 77.4600),
    ("Himachal Pradesh", "Kinnaur"): (31.5900, 78.4100), ("Himachal Pradesh", "Lahaul Spiti"): (32.5700, 77.5700),
    ("Uttarakhand", "Dehradun"): (30.3165, 78.0322), ("Uttarakhand", "Haridwar"): (29.9457, 78.1642),
    ("Uttarakhand", "Nainital"): (29.3803, 79.4636), ("Uttarakhand", "Udham Singh Nagar"): (28.9833, 79.5167),
    ("Uttarakhand", "Pauri Garhwal"): (29.7000, 78.7800), ("Uttarakhand", "Tehri Garhwal"): (30.3780, 78.4804),
    ("Uttarakhand", "Chamoli"): (30.4024, 79.3193), ("Uttarakhand", "Almora"): (29.5971, 79.6590),
    ("Uttarakhand", "Pithoragarh"): (29.5826, 80.2180), ("Uttarakhand", "Champawat"): (29.3339, 80.0910),
    ("Uttarakhand", "Bageshwar"): (29.8372, 79.7714), ("Uttarakhand", "Rudraprayag"): (30.2844, 78.9819),
    ("Uttarakhand", "Uttarkashi"): (30.7268, 78.4354),
    ("Goa", "North Goa"): (15.4909, 73.8278), ("Goa", "South Goa"): (15.1734, 74.0573),
    ("Delhi", "New Delhi"): (28.6139, 77.2090), ("Delhi", "Central Delhi"): (28.6508, 77.2295),
    ("Delhi", "East Delhi"): (28.6600, 77.3100), ("Delhi", "North Delhi"): (28.7200, 77.2000),
    ("Delhi", "South Delhi"): (28.5400, 77.2200), ("Delhi", "West Delhi"): (28.6500, 77.1000),
    ("Delhi", "North East Delhi"): (28.7000, 77.3000), ("Delhi", "North West Delhi"): (28.7000, 77.1000),
    ("Delhi", "South East Delhi"): (28.5700, 77.2900), ("Delhi", "South West Delhi"): (28.5500, 77.0800),
    ("Delhi", "Shahdara"): (28.6700, 77.3000),
    ("Chandigarh", "Chandigarh"): (30.7333, 76.7794),
    ("Puducherry", "Puducherry"): (11.9416, 79.8083), ("Puducherry", "Karaikal"): (10.9254, 79.8380),
    ("Puducherry", "Mahe"): (11.7010, 75.5362), ("Puducherry", "Yanam"): (16.7300, 82.2130),
    ("Jammu and Kashmir", "Srinagar"): (34.0837, 74.7973), ("Jammu and Kashmir", "Jammu"): (32.7266, 74.8570),
    ("Jammu and Kashmir", "Anantnag"): (33.7311, 75.1487), ("Jammu and Kashmir", "Baramulla"): (34.2090, 74.3442),
    ("Jammu and Kashmir", "Pulwama"): (33.8742, 74.8985), ("Jammu and Kashmir", "Kupwara"): (34.5211, 74.2615),
    ("Jammu and Kashmir", "Rajouri"): (33.3771, 74.3102), ("Jammu and Kashmir", "Udhampur"): (32.9160, 75.1410),
    ("Jammu and Kashmir", "Kathua"): (32.3842, 75.5160), ("Jammu and Kashmir", "Poonch"): (33.7726, 74.0927),
    ("Jammu and Kashmir", "Budgam"): (33.9400, 74.7100), ("Jammu and Kashmir", "Kulgam"): (33.6440, 75.0190),
    ("Jammu and Kashmir", "Shopian"): (33.7200, 74.8300), ("Jammu and Kashmir", "Ganderbal"): (34.2200, 74.7700),
    ("Jammu and Kashmir", "Bandipora"): (34.4100, 74.6400), ("Jammu and Kashmir", "Reasi"): (33.0800, 74.8300),
    ("Jammu and Kashmir", "Ramban"): (33.2400, 75.2400), ("Jammu and Kashmir", "Doda"): (33.1500, 75.5500),
    ("Jammu and Kashmir", "Kishtwar"): (33.3100, 75.7700), ("Jammu and Kashmir", "Samba"): (32.5800, 75.1200),
    ("Ladakh", "Leh"): (34.1526, 77.5771), ("Ladakh", "Kargil"): (34.5539, 76.1349),
    ("Sikkim", "East Sikkim"): (27.3389, 88.6065), ("Sikkim", "West Sikkim"): (27.2900, 88.2600),
    ("Sikkim", "North Sikkim"): (27.6700, 88.4500), ("Sikkim", "South Sikkim"): (27.1500, 88.5300),
    ("Sikkim", "Pakyong"): (27.2300, 88.6100), ("Sikkim", "Soreng"): (27.2000, 88.1000),
    ("Manipur", "Imphal East"): (24.8170, 93.9368), ("Manipur", "Imphal West"): (24.8000, 93.9400),
    ("Manipur", "Bishnupur"): (24.6100, 93.7700), ("Manipur", "Thoubal"): (24.6300, 94.0100),
    ("Manipur", "Chandel"): (24.3300, 94.0200), ("Manipur", "Senapati"): (25.2700, 94.0200),
    ("Manipur", "Tamenglong"): (24.9800, 93.4800), ("Manipur", "Churachandpur"): (24.3300, 93.6800),
    ("Manipur", "Ukhrul"): (25.1200, 94.3600), ("Manipur", "Jiribam"): (24.8000, 93.1200),
    ("Manipur", "Kakching"): (24.5000, 93.9900), ("Manipur", "Kamjong"): (25.1100, 94.6200),
    ("Manipur", "Kangpokpi"): (25.1300, 93.9700), ("Manipur", "Noney"): (25.0000, 93.8000),
    ("Manipur", "Pherzawl"): (24.0700, 93.4700), ("Manipur", "Tengnoupal"): (24.0100, 94.1800),
    ("Meghalaya", "East Khasi Hills"): (25.5788, 91.8933), ("Meghalaya", "West Khasi Hills"): (25.3500, 91.2700),
    ("Meghalaya", "East Garo Hills"): (25.6200, 90.4900), ("Meghalaya", "West Garo Hills"): (25.5700, 90.2200),
    ("Meghalaya", "Ri Bhoi"): (25.7300, 92.0200), ("Meghalaya", "East Jaintia Hills"): (25.3600, 92.5100),
    ("Meghalaya", "South Garo Hills"): (25.0300, 90.4200), ("Meghalaya", "North Garo Hills"): (25.8700, 90.5700),
    ("Meghalaya", "South West Garo Hills"): (25.2900, 89.9800), ("Meghalaya", "South West Khasi Hills"): (25.1100, 91.2600),
    ("Meghalaya", "Eastern West Khasi Hills"): (25.4500, 91.5600),
    ("Tripura", "West Tripura"): (23.8315, 91.2868), ("Tripura", "North Tripura"): (24.3600, 92.0000),
    ("Tripura", "South Tripura"): (23.2700, 91.7500), ("Tripura", "Gomati"): (23.4500, 91.8400),
    ("Tripura", "Dhalai"): (24.0600, 92.0200), ("Tripura", "Khowai"): (24.0700, 91.6000),
    ("Tripura", "Sepahijala"): (23.6700, 91.3000), ("Tripura", "Unakoti"): (24.3300, 92.0600),
    ("Tripura", "Sipahijala"): (23.6700, 91.3000),
    ("Nagaland", "Kohima"): (25.6751, 94.1086), ("Nagaland", "Dimapur"): (25.9100, 93.7200),
    ("Nagaland", "Mokokchung"): (26.3200, 94.5200), ("Nagaland", "Wokha"): (26.1000, 94.2600),
    ("Nagaland", "Zunheboto"): (26.0000, 94.5200), ("Nagaland", "Tuensang"): (26.2700, 94.8200),
    ("Nagaland", "Mon"): (26.7200, 95.0100), ("Nagaland", "Phek"): (25.6500, 94.4700),
    ("Nagaland", "Kiphire"): (25.8500, 95.0200), ("Nagaland", "Longleng"): (26.5000, 94.5800),
    ("Nagaland", "Peren"): (25.4800, 93.7100), ("Nagaland", "Chumoukedima"): (25.8500, 93.7200),
    ("Nagaland", "Niuland"): (25.7000, 93.9800), ("Nagaland", "Noklak"): (26.1700, 95.2200),
    ("Nagaland", "Shamator"): (26.4000, 94.9000), ("Nagaland", "Tseminyu"): (25.9700, 94.1100),
    ("Arunachal Pradesh", "Tawang"): (27.5860, 91.8590), ("Arunachal Pradesh", "West Kameng"): (27.2200, 92.5600),
    ("Arunachal Pradesh", "East Kameng"): (27.0400, 93.0400), ("Arunachal Pradesh", "Papum Pare"): (27.1000, 93.6000),
    ("Arunachal Pradesh", "Kurung Kumey"): (28.0700, 93.8300), ("Arunachal Pradesh", "Kra Daadi"): (28.1700, 94.3700),
    ("Arunachal Pradesh", "Lower Subansiri"): (27.5600, 93.8700), ("Arunachal Pradesh", "Upper Subansiri"): (28.3000, 94.0800),
    ("Arunachal Pradesh", "West Siang"): (28.1600, 94.5700), ("Arunachal Pradesh", "East Siang"): (28.0900, 95.2000),
    ("Arunachal Pradesh", "Siang"): (28.0200, 94.9000), ("Arunachal Pradesh", "Upper Siang"): (28.7600, 95.1300),
    ("Arunachal Pradesh", "Lower Siang"): (27.9500, 94.6000), ("Arunachal Pradesh", "Lohit"): (27.8300, 96.3400),
    ("Arunachal Pradesh", "Anjaw"): (28.0600, 96.8400), ("Arunachal Pradesh", "Tirap"): (27.0200, 95.7600),
    ("Arunachal Pradesh", "Changlang"): (27.1300, 95.7400), ("Arunachal Pradesh", "Longding"): (27.3400, 95.6500),
    ("Arunachal Pradesh", "Namsai"): (27.6700, 95.8300), ("Arunachal Pradesh", "Dibang Valley"): (28.6300, 95.7200),
    ("Arunachal Pradesh", "Lower Dibang Valley"): (28.0700, 95.8300), ("Arunachal Pradesh", "Lepa Rada"): (27.9800, 94.7200),
    ("Arunachal Pradesh", "Shi Yomi"): (28.3900, 94.7100), ("Arunachal Pradesh", "Pakke Kessang"): (27.1700, 93.5200),
    ("Arunachal Pradesh", "Kamle"): (27.7500, 93.5900),
    ("Mizoram", "Aizawl"): (23.7307, 92.7173), ("Mizoram", "Lunglei"): (22.8800, 92.7300),
    ("Mizoram", "Champhai"): (23.4600, 93.3200), ("Mizoram", "Kolasib"): (24.2300, 92.6800),
    ("Mizoram", "Serchhip"): (23.3100, 92.8500), ("Mizoram", "Mamit"): (23.9300, 92.4800),
    ("Mizoram", "Lawngtlai"): (22.0300, 92.9000), ("Mizoram", "Siaha"): (22.4800, 92.9700),
    ("Mizoram", "Hnahthial"): (23.0000, 92.7200), ("Mizoram", "Khawzawl"): (23.3000, 93.0200),
    ("Mizoram", "Saitual"): (23.7900, 92.9900),
    ("Andaman and Nicobar Islands", "South Andaman"): (11.6234, 92.7265),
    ("Andaman and Nicobar Islands", "North and Middle Andaman"): (12.5800, 92.8500),
    ("Andaman and Nicobar Islands", "Nicobar"): (8.0883, 93.7760),
    ("Lakshadweep", "Lakshadweep"): (10.5667, 72.6417),
    ("Dadra and Nagar Haveli and Daman and Diu", "Dadra and Nagar Haveli"): (20.1809, 73.0169),
    ("Dadra and Nagar Haveli and Daman and Diu", "Daman"): (20.3974, 72.8328),
    ("Dadra and Nagar Haveli and Daman and Diu", "Diu"): (20.7144, 70.9874),
}

INDIA_STATES_DISTRICTS = {
    "Andhra Pradesh": ["Alluri Sitharama Raju","Anakapalli","Anantapur","Annamayya","Bapatla","Chittoor","East Godavari","Eluru","Guntur","Kakinada","Konaseema","Krishna","Kurnool","Nandyal","Nellore","NTR","Palnadu","Parvathipuram Manyam","Prakasam","Sri Balaji","Sri Sathya Sai","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari","YSR Kadapa"],
    "Telangana": ["Adilabad","Bhadradri Kothagudem","Hanamkonda","Hyderabad","Jagtial","Jangaon","Jayashankar Bhupalpally","Jogulamba Gadwal","Kamareddy","Karimnagar","Khammam","Komaram Bheem","Mahabubabad","Mahabubnagar","Mancherial","Medak","Medchal Malkajgiri","Mulugu","Nagarkurnool","Nalgonda","Narayanpet","Nirmal","Nizamabad","Peddapalli","Rajanna Sircilla","Rangareddy","Sangareddy","Siddipet","Suryapet","Vikarabad","Wanaparthy","Warangal","Yadadri Bhuvanagiri"],
    "Maharashtra": ["Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur","Dhule","Gadchiroli","Gondia","Hingoli","Jalgaon","Jalna","Kolhapur","Latur","Mumbai City","Mumbai Suburban","Nagpur","Nanded","Nandurbar","Nashik","Osmanabad","Palghar","Parbhani","Pune","Raigad","Ratnagiri","Sangli","Satara","Sindhudurg","Solapur","Thane","Wardha","Washim","Yavatmal"],
    "Karnataka": ["Bagalkot","Ballari","Belagavi","Bengaluru Rural","Bengaluru Urban","Bidar","Chamarajanagar","Chikkaballapur","Chikkamagaluru","Chitradurga","Dakshina Kannada","Davanagere","Dharwad","Gadag","Hassan","Haveri","Kalaburagi","Kodagu","Kolar","Koppal","Mandya","Mysuru","Raichur","Ramanagara","Shivamogga","Tumakuru","Udupi","Uttara Kannada","Vijayapura","Yadgir"],
    "Tamil Nadu": ["Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri","Dindigul","Erode","Kallakurichi","Kancheepuram","Kanyakumari","Karur","Krishnagiri","Madurai","Mayiladuthurai","Nagapattinam","Namakkal","Nilgiris","Perambalur","Pudukkottai","Ramanathapuram","Ranipet","Salem","Sivaganga","Tenkasi","Thanjavur","Theni","Thoothukudi","Tiruchirappalli","Tirunelveli","Tirupathur","Tiruppur","Tiruvallur","Tiruvannamalai","Tiruvarur","Vellore","Viluppuram","Virudhunagar"],
    "Punjab": ["Amritsar","Barnala","Bathinda","Faridkot","Fatehgarh Sahib","Fazilka","Ferozepur","Gurdaspur","Hoshiarpur","Jalandhar","Kapurthala","Ludhiana","Mansa","Moga","Mohali","Muktsar","Pathankot","Patiala","Rupnagar","Sangrur","Shaheed Bhagat Singh Nagar","Tarn Taran"],
    "Haryana": ["Ambala","Bhiwani","Charkhi Dadri","Faridabad","Fatehabad","Gurugram","Hisar","Jhajjar","Jind","Kaithal","Karnal","Kurukshetra","Mahendragarh","Nuh","Palwal","Panchkula","Panipat","Rewari","Rohtak","Sirsa","Sonipat","Yamunanagar"],
    "Uttar Pradesh": ["Agra","Aligarh","Ambedkar Nagar","Amethi","Amroha","Auraiya","Ayodhya","Azamgarh","Baghpat","Bahraich","Ballia","Balrampur","Banda","Barabanki","Bareilly","Basti","Bhadohi","Bijnor","Budaun","Bulandshahr","Chandauli","Chitrakoot","Deoria","Etah","Etawah","Farrukhabad","Fatehpur","Firozabad","Gautam Buddha Nagar","Ghaziabad","Ghazipur","Gonda","Gorakhpur","Hamirpur","Hapur","Hardoi","Hathras","Jalaun","Jaunpur","Jhansi","Kannauj","Kanpur Dehat","Kanpur Nagar","Kasganj","Kaushambi","Kushinagar","Lakhimpur Kheri","Lalitpur","Lucknow","Maharajganj","Mahoba","Mainpuri","Mathura","Mau","Meerut","Mirzapur","Moradabad","Muzaffarnagar","Pilibhit","Pratapgarh","Prayagraj","Rae Bareli","Rampur","Saharanpur","Sambhal","Sant Kabir Nagar","Shahjahanpur","Shamli","Shravasti","Siddharthnagar","Sitapur","Sonbhadra","Sultanpur","Unnao","Varanasi"],
    "Madhya Pradesh": ["Agar Malwa","Alirajpur","Anuppur","Ashoknagar","Balaghat","Barwani","Betul","Bhind","Bhopal","Burhanpur","Chhatarpur","Chhindwara","Damoh","Datia","Dewas","Dhar","Dindori","Guna","Gwalior","Harda","Hoshangabad","Indore","Jabalpur","Jhabua","Katni","Khandwa","Khargone","Mandla","Mandsaur","Morena","Narsinghpur","Neemuch","Niwari","Panna","Raisen","Rajgarh","Ratlam","Rewa","Sagar","Satna","Sehore","Seoni","Shahdol","Shajapur","Sheopur","Shivpuri","Sidhi","Singrauli","Tikamgarh","Ujjain","Umaria","Vidisha"],
    "Rajasthan": ["Ajmer","Alwar","Banswara","Baran","Barmer","Bharatpur","Bhilwara","Bikaner","Bundi","Chittorgarh","Churu","Dausa","Dholpur","Dungarpur","Hanumangarh","Jaipur","Jaisalmer","Jalore","Jhalawar","Jhunjhunu","Jodhpur","Karauli","Kota","Nagaur","Pali","Pratapgarh","Rajsamand","Sawai Madhopur","Sikar","Sirohi","Sri Ganganagar","Tonk","Udaipur"],
    "Gujarat": ["Ahmedabad","Amreli","Anand","Aravalli","Banaskantha","Bharuch","Bhavnagar","Botad","Chhota Udaipur","Dahod","Dang","Devbhoomi Dwarka","Gandhinagar","Gir Somnath","Jamnagar","Junagadh","Kheda","Kutch","Mahisagar","Mehsana","Morbi","Narmada","Navsari","Panchmahal","Patan","Porbandar","Rajkot","Sabarkantha","Surat","Surendranagar","Tapi","Vadodara","Valsad"],
    "Bihar": ["Araria","Arwal","Aurangabad","Banka","Begusarai","Bhagalpur","Bhojpur","Buxar","Darbhanga","East Champaran","Gaya","Gopalganj","Jamui","Jehanabad","Kaimur","Katihar","Khagaria","Kishanganj","Lakhisarai","Madhepura","Madhubani","Munger","Muzaffarpur","Nalanda","Nawada","Patna","Purnia","Rohtas","Saharsa","Samastipur","Saran","Sheikhpura","Sheohar","Sitamarhi","Siwan","Supaul","Vaishali","West Champaran"],
    "West Bengal": ["Alipurduar","Bankura","Birbhum","Cooch Behar","Dakshin Dinajpur","Darjeeling","Hooghly","Howrah","Jalpaiguri","Jhargram","Kalimpong","Kolkata","Malda","Murshidabad","Nadia","North 24 Parganas","Paschim Bardhaman","Paschim Medinipur","Purba Bardhaman","Purba Medinipur","Purulia","South 24 Parganas","Uttar Dinajpur"],
    "Odisha": ["Angul","Balangir","Balasore","Bargarh","Bhadrak","Boudh","Cuttack","Deogarh","Dhenkanal","Gajapati","Ganjam","Jagatsinghpur","Jajpur","Jharsuguda","Kalahandi","Kandhamal","Kendrapara","Kendujhar","Khordha","Koraput","Malkangiri","Mayurbhanj","Nabarangpur","Nayagarh","Nuapada","Puri","Rayagada","Sambalpur","Subarnapur","Sundargarh"],
    "Assam": ["Bajali","Baksa","Barpeta","Biswanath","Bongaigaon","Cachar","Charaideo","Chirang","Darrang","Dhemaji","Dhubri","Dibrugarh","Dima Hasao","Goalpara","Golaghat","Hailakandi","Hojai","Jorhat","Kamrup","Kamrup Metropolitan","Karbi Anglong","Karimganj","Kokrajhar","Lakhimpur","Majuli","Morigaon","Nagaon","Nalbari","Sivasagar","Sonitpur","South Salmara","Tinsukia","Udalguri","West Karbi Anglong"],
    "Himachal Pradesh": ["Bilaspur","Chamba","Hamirpur","Kangra","Kinnaur","Kullu","Lahaul Spiti","Mandi","Shimla","Sirmaur","Solan","Una"],
    "Uttarakhand": ["Almora","Bageshwar","Chamoli","Champawat","Dehradun","Haridwar","Nainital","Pauri Garhwal","Pithoragarh","Rudraprayag","Tehri Garhwal","Udham Singh Nagar","Uttarkashi"],
    "Jharkhand": ["Bokaro","Chatra","Deoghar","Dhanbad","Dumka","East Singhbhum","Garhwa","Giridih","Godda","Gumla","Hazaribagh","Jamtara","Khunti","Koderma","Latehar","Lohardaga","Pakur","Palamu","Ramgarh","Ranchi","Sahebganj","Seraikela Kharsawan","Simdega","West Singhbhum"],
    "Chhattisgarh": ["Balod","Baloda Bazar","Balrampur","Bastar","Bemetara","Bijapur","Bilaspur","Dantewada","Dhamtari","Durg","Gariaband","Gaurela Pendra Marwahi","Janjgir Champa","Jashpur","Kabirdham","Kanker","Khairagarh","Kondagaon","Korba","Koriya","Mahasamund","Manendragarh","Mohla Manpur","Mungeli","Narayanpur","Raigarh","Raipur","Rajnandgaon","Sakti","Sarangarh Bilaigarh","Sukma","Surajpur","Surguja"],
    "Kerala": ["Alappuzha","Ernakulam","Idukki","Kannur","Kasaragod","Kollam","Kottayam","Kozhikode","Malappuram","Palakkad","Pathanamthitta","Thiruvananthapuram","Thrissur","Wayanad"],
    "Goa": ["North Goa","South Goa"],
    "Manipur": ["Bishnupur","Chandel","Churachandpur","Imphal East","Imphal West","Jiribam","Kakching","Kamjong","Kangpokpi","Noney","Pherzawl","Senapati","Tamenglong","Tengnoupal","Thoubal","Ukhrul"],
    "Meghalaya": ["East Garo Hills","East Jaintia Hills","East Khasi Hills","Eastern West Khasi Hills","North Garo Hills","Ri Bhoi","South Garo Hills","South West Garo Hills","South West Khasi Hills","West Garo Hills","West Jaintia Hills","West Khasi Hills"],
    "Tripura": ["Dhalai","Gomati","Khowai","North Tripura","Sepahijala","South Tripura","Sipahijala","Unakoti","West Tripura"],
    "Nagaland": ["Chumoukedima","Dimapur","Kiphire","Kohima","Longleng","Mokokchung","Mon","Niuland","Noklak","Peren","Phek","Shamator","Tseminyu","Tuensang","Wokha","Zunheboto"],
    "Arunachal Pradesh": ["Anjaw","Changlang","Dibang Valley","East Kameng","East Siang","Kamle","Kra Daadi","Kurung Kumey","Lepa Rada","Lohit","Longding","Lower Dibang Valley","Lower Siang","Lower Subansiri","Namsai","Pakke Kessang","Papum Pare","Shi Yomi","Siang","Tawang","Tirap","Upper Siang","Upper Subansiri","West Kameng","West Siang"],
    "Mizoram": ["Aizawl","Champhai","Hnahthial","Khawzawl","Kolasib","Lawngtlai","Lunglei","Mamit","Saitual","Serchhip","Siaha"],
    "Sikkim": ["East Sikkim","North Sikkim","Pakyong","Soreng","South Sikkim","West Sikkim"],
    "Jammu and Kashmir": ["Anantnag","Bandipora","Baramulla","Budgam","Doda","Ganderbal","Jammu","Kathua","Kishtwar","Kulgam","Kupwara","Poonch","Pulwama","Rajouri","Ramban","Reasi","Samba","Shopian","Srinagar","Udhampur"],
    "Ladakh": ["Kargil","Leh"],
    "Puducherry": ["Karaikal","Mahe","Puducherry","Yanam"],
    "Chandigarh": ["Chandigarh"],
    "Delhi": ["Central Delhi","East Delhi","New Delhi","North Delhi","North East Delhi","North West Delhi","Shahdara","South Delhi","South East Delhi","South West Delhi","West Delhi"],
    "Andaman and Nicobar Islands": ["Nicobar","North and Middle Andaman","South Andaman"],
    "Lakshadweep": ["Lakshadweep"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Dadra and Nagar Haveli","Daman","Diu"],
}

def get_climate_data(village, district, state):
    try:
        coords = DISTRICT_COORDS.get((state, district))
        if not coords:
            return None, f"Coordinates not found for {district}, {state}"
        lat, lon = coords

        if village and village.strip():
            try:
                geo_url = (
                    "https://geocoding-api.open-meteo.com"
                    "/v1/search"
                    f"?name={requests.utils.quote(village.strip())}"
                    "&count=5&language=en&format=json"
                )
                geo_resp = requests.get(geo_url, timeout=5)
                geo_data = geo_resp.json()
                results = geo_data.get("results", [])
                india_results = [r for r in results if r.get("country_code", "").upper() == "IN"]
                state_results = [r for r in india_results if state.lower() in r.get("admin1", "").lower()]
                if state_results:
                    lat = state_results[0]["latitude"]
                    lon = state_results[0]["longitude"]
                    location_label = f"{village}, {district}, {state}"
                    note = "Village location found ✓"
                else:
                    location_label = f"{village}, {district}, {state}"
                    note = f"Using {district} district coordinates"
            except Exception:
                location_label = f"{village}, {district}, {state}"
                note = f"Using {district} district coordinates"
        else:
            location_label = f"{district}, {state}"
            note = "District coordinates used"

        climate_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            "&start_date=2014-01-01&end_date=2023-12-31"
            "&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            "&timezone=Asia%2FKolkata"
        )
        climate_resp = requests.get(climate_url, timeout=30)
        climate_data = climate_resp.json()
        daily = climate_data.get("daily", {})

        temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        hums  = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
        rains = [r for r in daily.get("precipitation_sum", []) if r is not None]

        avg_temp    = round(sum(temps) / len(temps), 1) if temps else 25.0
        avg_hum     = round(sum(hums) / len(hums), 1) if hums else 60.0
        annual_rain = round(sum(rains) / (len(rains) / 365), 1) if rains else 1000.0

        return {
            "location":    location_label,
            "note":        note,
            "temperature": avg_temp,
            "humidity":    avg_hum,
            "rainfall":    annual_rain,
        }, None

    except Exception as e:
        return None, f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════
# UI CONSTANTS
# ══════════════════════════════════════════════════════════════

SOIL_COLORS = {
    "Red Soil":      "#C62828",
    "Black Soil":    "#212121",
    "Alluvial Soil": "#795548",
    "Clay Soil":     "#FF8F00",
    "Laterite Soil": "#BF360C",
    "Yellow Soil":   "#F9A825",
    "Sandy Soil":    "#F4E409",
}

CROP_ICONS = {
    "Cotton":       "🌿", "Maize":      "🌽", "Rice":       "🌾",
    "Wheat":        "🌾", "Sugarcane":  "🎋", "Potato":     "🥔",
    "Tomato":       "🍅", "Groundnut":  "🥜", "Soybean":    "🫘",
    "Sunflower":    "🌻", "Barley":     "🌾", "Mustard":    "🌼",
    "Chickpea":     "🫘", "Watermelon": "🍉", "Cucumber":   "🥒",
    "Pumpkin":      "🎃", "Mango":      "🥭", "Banana":     "🍌",
    "Cashew":       "🌰", "Rubber":     "🌳", "Tea":        "🍵",
    "Coffee":       "☕", "Tapioca":    "🌿", "Turmeric":   "🟡",
    "Ginger":       "🫚", "Pineapple":  "🍍", "Jackfruit":  "🍈",
    "Jute":         "🌿", "Sorghum":    "🌾", "Sesame":     "🌿",
    "Linseed":      "🌼", "Safflower":  "🌼", "Moong":      "🫘",
    "Taro":         "🌿", "Spinach":    "🥬", "Muskmelon":  "🍈",
}


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

# ── Header + theme toggle ───────────────────────────────────────
_hcol1, _hcol2 = st.columns([8, 1])
with _hcol1:
    st.markdown("""
    <div class="hero" style="background:#1B5E20; color:white; padding:1.5rem 2rem;
    border-radius:16px; margin-bottom:1rem; text-align:center">
      <h1 style="margin:0; font-size:1.8rem; font-weight:800; letter-spacing:-0.5px; color:white !important">
        🌱 Multimodal Crop &amp; Fertilizer Recommendation</h1>
      <p style="margin:0.4rem 0 0; font-size:0.95rem; opacity:0.85; color:white !important">
        AI-powered soil analysis for smarter farming</p>
    </div>
    """, unsafe_allow_html=True)
with _hcol2:
    st.markdown("<div style='padding-top:12px'></div>", unsafe_allow_html=True)
    _btn_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    if st.button(_btn_label, key="theme_toggle"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

# ── Load models ────────────────────────────────────────────────
try:
    img_model, tab_proj, fusion, xgb_clf, scaler, CLASS_NAMES, NUMERIC_COLS = load_all_models()
    _models_ok = True
except Exception as _load_err:
    _models_ok = False
    st.error(
        f"**Model loading failed:** {_load_err}\n\n"
        f"This usually means Git LFS files were not pulled. "
        f"Run `git lfs pull` in the repo and redeploy."
    )
    st.markdown("**File status at startup:**")
    for _fn, _info in _file_status.items():
        _ok  = _info["exists"] and _info["mb"] > 0.1
        _ico = "OK" if _ok else "MISSING / TOO SMALL"
        st.write(f"- `{_fn}`: {_info['mb']:.1f} MB  —  {_ico}")
    st.stop()

if "auto_temp" not in st.session_state:
    st.session_state.auto_temp = 25.7
if "auto_hum" not in st.session_state:
    st.session_state.auto_hum = 58.2
if "auto_rain" not in st.session_state:
    st.session_state.auto_rain = 1619.0
if "location_name" not in st.session_state:
    st.session_state.location_name = ""
if "location_note" not in st.session_state:
    st.session_state.location_note = ""

left, right = st.columns([1, 1], gap="large")

# ── LEFT: Inputs ───────────────────────────────────────────────
with left:

    # Section: Soil Image
    st.markdown("""
    <div class="section-card">
    <h4 class="section-title">📷 Soil Image</h4>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload soil photo (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.info("💡 Tip: Upload a clear close-up photo of soil for best results. "
            "Avoid photos with people, plants, or bright objects.")
    if uploaded:
        st.session_state.img_bytes = uploaded.getvalue()
        st.session_state.last_result = None
        st.session_state.last_error = None
    img_bytes = st.session_state.img_bytes
    if img_bytes:
        st.image(io.BytesIO(img_bytes), use_container_width=True)

    # Section: Soil Chemical Properties
    st.markdown("""
    <div class="section-card" style="margin:16px 0 8px 0">
    <h4 class="section-title">🧪 Soil Chemical Properties</h4>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        n  = st.number_input("Nitrogen - N", 0.0, 200.0, 90.0, step=1.0,
                             help="kg/ha (kilograms per hectare) | Range: 0-140")
    with c2:
        p  = st.number_input("Phosphorus - P", 0.0, 200.0, 42.0, step=1.0,
                             help="kg/ha (kilograms per hectare) | Range: 0-145")
    with c3:
        k  = st.number_input("Potassium - K", 0.0, 200.0, 43.0, step=1.0,
                             help="kg/ha (kilograms per hectare) | Range: 0-205")
    ph = st.number_input("Soil pH", 3.0, 10.0, 6.5, step=0.01,
                         help="Acidity level | Range: 3.5 to 9.5")

    # Section: Environmental Conditions
    st.markdown("""
    <div class="section-card" style="margin:16px 0 8px 0">
    <h4 class="section-title">🌡️ Environmental Conditions</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#E8F5E9; border-left:4px solid #2E7D32;
    border-radius:12px; padding:16px; margin-bottom:12px">
    <h4 style="color:#1B5E20; margin:0 0 4px 0">🌍 Auto-Fill Climate Data</h4>
    <p style="color:#555; font-size:13px; margin:0">
    Select your location to auto-fill Temperature, Humidity and Rainfall
    </p></div>
    """, unsafe_allow_html=True)

    sel_state = st.selectbox(
        "📍 Select Your State",
        options=["-- Select State --"] + sorted(INDIA_STATES_DISTRICTS.keys()),
        index=0,
    )

    if sel_state and sel_state != "-- Select State --":
        sel_district = st.selectbox(
            "🏛️ Select Your District",
            options=["-- Select District --"] + INDIA_STATES_DISTRICTS[sel_state],
            index=0,
        )
    else:
        sel_district = "-- Select District --"
        st.selectbox("🏛️ Select Your District", options=["-- Select State First --"], disabled=True)

    loc3, loc4 = st.columns([3, 1])
    with loc3:
        village = st.text_input("🏘️ Enter Village / Town Name", placeholder="e.g. Kodad, Suryapet, Miryalaguda, Nalgonda...")
    with loc4:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🌤️ Fetch", use_container_width=True)

    if fetch_btn:
        if sel_state == "-- Select State --":
            st.warning("⚠️ Please select your state")
        elif sel_district == "-- Select District --":
            st.warning("⚠️ Please select your district")
        else:
            with st.spinner("Fetching climate data..."):
                climate, error = get_climate_data(village, sel_district, sel_state)
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state.auto_temp = climate["temperature"]
                st.session_state.auto_hum  = climate["humidity"]
                st.session_state.auto_rain = climate["rainfall"]
                st.session_state.location_name = climate["location"]
                st.session_state.location_note = climate.get("note", "")
                st.success(
                    f"✅ {climate['location']} — "
                    f"{climate.get('note', 'Data loaded')}"
                )
                st.rerun()

    if st.session_state.location_name:
        st.markdown(f"""
        <div style="background:#F1F8E9; border-radius:10px; padding:12px 16px;
        margin:8px 0; border:1px solid #C8E6C9">
        <p style="margin:0; font-size:13px; color:#1B5E20">
        <strong>📍 {st.session_state.location_name}</strong></p>
        <p style="margin:4px 0 0; font-size:12px; color:#888">
        ℹ️ {st.session_state.location_note}
        </p>
        <p style="margin:6px 0 0; font-size:13px; color:#555">
        🌡️ <strong>{st.session_state.auto_temp}°C</strong> &nbsp;|&nbsp;
        💧 <strong>{st.session_state.auto_hum}%</strong> &nbsp;|&nbsp;
        🌧️ <strong>{st.session_state.auto_rain}mm</strong>
        </p>
        <p style="margin:4px 0 0; font-size:11px; color:#aaa">
        10-year averages (2014-2023) • Values auto-filled below ↓
        </p></div>
        """, unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        temp = st.number_input("Temperature", 10.0, 45.0,
                               float(st.session_state.auto_temp), step=0.1,
                               help="°C (Celsius) | Range: 10-45°C | Auto-filled from location")
    with e2:
        hum  = st.number_input("Humidity", 14.0, 100.0,
                               float(st.session_state.auto_hum), step=0.1,
                               help="% Relative humidity | Range: 14-100% | Auto-filled from location")
    with e3:
        rain = st.number_input("Rainfall", 200.0, 3000.0,
                               float(st.session_state.auto_rain), step=1.0,
                               help="mm per year | Range: 20-2500 mm | Auto-filled from location")

    # Section: Farm History
    st.markdown("""
    <div class="section-card" style="margin:16px 0 8px 0">
    <h4 class="section-title">📊 Farm History</h4>
    </div>
    """, unsafe_allow_html=True)
    st.info("""💡 **Unit Guide for Indian Farmers:**
- NPK & Fertilizer: kg/ha (1 acre = 0.4 hectare)
- To convert: multiply your per-acre value × 2.47 to get per-hectare value
- Yield: t/ha (tonnes per hectare)""")
    h1, h2 = st.columns(2)
    with h1:
        yld  = st.number_input("Yield Last Season", 0.0, 15000.0, 2500.0, step=10.0,
                               help="t/ha = tonnes per hectare | 1 acre ≈ 0.4 ha | Range: 0.5-10 t/ha")
    with h2:
        fert = st.number_input("Fertilizer Used", 0.0, 1000.0, 120.0, step=5.0,
                               help="kg/ha = kilograms per hectare | 1 acre ≈ 0.4 ha | Range: 50-500 kg/ha")

    # Section: Farm Details
    st.markdown("""
    <div class="section-card" style="margin:16px 0 8px 0">
    <h4 class="section-title">🌾 Farm Details</h4>
    </div>
    """, unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        prev   = st.selectbox("Previous Crop",
                               ["Wheat","Rice","Maize","Cotton","Potato","Sugarcane","Tomato"])
    with d2:
        irrig  = st.selectbox("Irrigation", ["Canal","Drip","Rainfed","Sprinkler"])
        region = st.selectbox("Region", ["South","North","East","West","Central"])

    predict_clicked = st.button("🔍 Analyze Soil")

# ── RIGHT: Results ─────────────────────────────────────────────
with right:
    if not predict_clicked and not st.session_state.last_result and not st.session_state.last_error:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem;
        color:#9E9E9E; border:2px dashed #C8E6C9; border-radius:16px;
        background:var(--card-bg,#FAFFFE)">
          <div style="font-size:3rem; margin-bottom:1rem">Soil</div>
          <p style="font-size:1.05rem; color:#555; margin:0">
            <strong>Upload a soil image</strong> and fill in the parameters,<br>
            then click <strong>Analyze Soil</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)

    if predict_clicked and not img_bytes:
        st.error("Please upload a soil image before analyzing.")

    if predict_clicked and img_bytes:
        _pil_check = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if not is_soil_image(_pil_check):
            st.markdown("""
            <div style="background:#FFEBEE;
            border-left:4px solid #C62828;
            border-radius:12px; padding:24px;
            margin:16px 0">
            <h3 style="color:#C62828; margin:0 0 8px 0">
            ❌ No Soil Detected</h3>
            <p style="color:#B71C1C; margin:0 0 12px 0">
            Please upload a clear soil photograph.</p>
            <p style="color:#555; font-size:14px; margin:0">
            ✓ Red, Black, Clay, Alluvial soil photos<br>
            ✓ Soil held in hands<br>
            ✓ Soil in containers or farm fields
            </p></div>
            """, unsafe_allow_html=True)
            st.stop()

        st.session_state.last_error = None
        with st.spinner("Running AI inference..."):
                try:
                    soil_name, confidence, all_probs, soil_fert, crop_recs, dbg = run_inference(
                        img_model, tab_proj, fusion, xgb_clf, scaler,
                        CLASS_NAMES, NUMERIC_COLS,
                        img_bytes,
                        n, p, k, temp, hum, rain, ph, yld, fert,
                        season, irrig, prev, region,
                    )
                    st.session_state.last_result = {
                        "soil_name": soil_name,
                        "confidence": confidence,
                        "all_probs": all_probs,
                        "soil_fert": soil_fert,
                        "crop_recs": crop_recs,
                        "dbg": dbg,
                    }
                except Exception as e:
                    st.session_state.last_error = f"Prediction failed: {e}"
                    st.session_state.last_result = None

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.last_result:
        res = st.session_state.last_result
        soil_name = res["soil_name"]
        confidence = res["confidence"]
        all_probs = res["all_probs"]
        soil_fert = res["soil_fert"]
        crop_recs = res["crop_recs"]
        dbg = res["dbg"]

        color = SOIL_COLORS.get(soil_name, "#2E7D32")
        st.markdown(f"""
        <div style="background:{color}; padding:24px;
        border-radius:16px; color:white; margin-bottom:16px">
          <p style="opacity:0.8; margin:0; font-size:13px;
          text-transform:uppercase; letter-spacing:2px; color:white">
          Detected Soil Type</p>
          <h1 style="margin:8px 0; font-size:36px; font-weight:800; color:white">
          {soil_name}</h1>
          <div style="background:rgba(255,255,255,0.2);
          display:inline-block; padding:6px 16px;
          border-radius:20px; font-size:14px; font-weight:600">
          Confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-weight:700; color:#1B5E20;
        font-size:15px; margin:16px 0 8px 0">
        Soil Probability Breakdown</p>
        """, unsafe_allow_html=True)
        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
        for soil, prob in sorted_probs:
            bar_color = SOIL_COLORS.get(soil, "#2E7D32")
            st.markdown(f"""
            <div style="display:flex; align-items:center;
            margin:6px 0; gap:10px">
              <span style="width:110px; font-size:13px;
              color:var(--prob-label,#333); flex-shrink:0">{soil}</span>
              <div style="flex:1; background:var(--prob-track,#f0f0f0);
              border-radius:6px; height:12px">
                <div style="width:{prob}%; background:{bar_color};
                height:12px; border-radius:6px"></div>
              </div>
              <span style="width:52px; text-align:right;
              font-size:13px; font-weight:600;
              color:{bar_color}">{prob:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <p style="font-weight:700; color:#1B5E20;
        font-size:15px; margin:0 0 8px 0">
        Recommended Crops</p>
        """, unsafe_allow_html=True)
        for crop in crop_recs:
            icon  = CROP_ICONS.get(crop["name"], "")
            stars = "?" * crop["stars"]
            st.markdown(f"""
            <div style="background:var(--card-bg,white); border:1px solid var(--card-border,#C8E6C9);
            border-radius:12px; padding:16px; margin:8px 0;
            border-left:4px solid #2E7D32">
              <div style="display:flex; align-items:center; gap:12px">
                <span style="font-size:32px">{icon}</span>
                <div>
                  <h4 style="margin:0; color:#1B5E20">
                  {crop["name"]}</h4>
                  <p style="margin:4px 0; color:#666; font-size:13px">
                  {stars} &nbsp;Rank #{crop["rank"]}</p>
                  <p style="margin:0; color:#2E7D32; font-size:13px">
                   {crop["fertilizer"]} &nbsp;|&nbsp;
                   {crop["npk"]}</p>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:var(--fert-bg,#FFF8E1); border-radius:12px;
        padding:20px; border-left:4px solid #FF8F00; margin-top:16px">
          <h4 style="color:#E65100; margin:0 0 12px 0">
           Fertilizer Recommendation</h4>
          <p style="margin:4px 0"><strong>Type:</strong>
          {soil_fert["fertilizer"]}</p>
          <p style="margin:4px 0"><strong>NPK Dosage:</strong>
          {soil_fert["npk"]} kg/ha</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("1 hectare = 2.47 acres | Divide kg/ha by 2.47 to get kg/acre")

        with st.expander("Prediction Debug"):
            st.write("Raw probs:", dbg["probs"])
            st.write("Image feat std:", dbg["img_feat_std"])

