# streamlit_app.py — Multimodal Soil & Crop Recommendation System
# ResNet-50 + XGBoost + TSACA Fusion + GRN  |  Accuracy: 98.67%
# Run: streamlit run streamlit_app.py

import io, os, json, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="SoilSense — Crop & Soil Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS theme ──────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }

  .stApp { background: #F9FBF9; }

  /* Header */
  .app-header {
    background: #2E7D32; color: white;
    padding: 1.2rem 1.8rem; border-radius: 10px;
    margin-bottom: 1.5rem;
  }
  .app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 800; }
  .app-header p  { margin: .2rem 0 0; font-size: .9rem; opacity: .85; }
  .acc-badge {
    display: inline-block; margin-top: .5rem;
    background: rgba(255,255,255,.2); border-radius: 20px;
    padding: .2rem .8rem; font-size: .78rem; font-weight: 600;
  }

  /* Section labels */
  .sec-label {
    font-size: .7rem; font-weight: 700; letter-spacing: 1px;
    color: #2E7D32; text-transform: uppercase;
    margin: 1.1rem 0 .4rem; border-left: 3px solid #2E7D32;
    padding-left: .5rem;
  }

  /* Soil result card */
  .soil-result {
    background: #2E7D32; color: white;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .soil-result .label { font-size: .65rem; opacity: .8; letter-spacing: 1px; text-transform: uppercase; }
  .soil-result .name  { font-size: 1.6rem; font-weight: 800; margin: .2rem 0; }
  .soil-result .conf  {
    display: inline-block; background: rgba(255,255,255,.2);
    border-radius: 6px; padding: .15rem .65rem;
    font-size: .9rem; font-weight: 600;
  }

  /* Probability rows */
  .prob-row {
    display: flex; align-items: center; gap: .6rem;
    margin-bottom: .35rem; font-size: .82rem; color: #212121;
  }
  .prob-label { width: 110px; flex-shrink: 0; }
  .prob-bar-bg { flex: 1; height: 8px; background: #E8F5E9; border-radius: 4px; overflow: hidden; }
  .prob-bar-fill { height: 100%; background: #2E7D32; border-radius: 4px; }
  .prob-pct { width: 38px; text-align: right; font-weight: 600; color: #2E7D32; }

  /* Crop cards */
  .crop-card {
    background: white; border: 1px solid #E8F5E9;
    border-radius: 8px; padding: .75rem 1rem;
    margin-bottom: .5rem;
    display: flex; align-items: center; gap: .8rem;
  }
  .crop-card.top { border-color: #2E7D32; background: #F1F8E9; }
  .crop-rank-badge {
    background: #2E7D32; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700; flex-shrink: 0;
  }
  .crop-rank-badge.s { background: #81C784; }
  .crop-rank-badge.t { background: #A5D6A7; color: #212121; }
  .crop-info .crop-name { font-weight: 700; font-size: .95rem; color: #212121; }
  .crop-info .crop-sub  { font-size: .75rem; color: #555; margin-top: .1rem; }
  .stars { color: #F9A825; font-size: .85rem; margin-left: auto; flex-shrink: 0; }

  /* Fertilizer table */
  .fert-box {
    background: #E8F5E9; border-left: 4px solid #2E7D32;
    border-radius: 0 8px 8px 0; padding: .9rem 1.1rem;
    margin-top: .75rem;
  }
  .fert-box .fert-title { font-size: .65rem; font-weight: 700;
    letter-spacing: 1px; color: #2E7D32; text-transform: uppercase; margin-bottom: .4rem; }
  .fert-box .fert-name  { font-size: 1rem; font-weight: 700; color: #212121; }
  .fert-box .fert-npk   { font-size: .85rem; color: #444; margin-top: .15rem; }

  /* Submit button */
  div[data-testid="stButton"] > button {
    background: #2E7D32; color: white; border: none;
    border-radius: 8px; padding: .65rem 1.5rem;
    font-size: .95rem; font-weight: 600; width: 100%;
  }
  div[data-testid="stButton"] > button:hover { background: #1B5E20; }

  /* Placeholder */
  .placeholder {
    text-align: center; padding: 4rem 1rem;
    color: #9E9E9E; border: 2px dashed #C8E6C9;
    border-radius: 10px;
  }
  .placeholder p { margin: .5rem 0 0; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

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
# UI
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <h1>SoilSense</h1>
  <p>Multimodal Crop &amp; Soil Recommendation System</p>
  <span class="acc-badge">Model Accuracy: 98.67%</span>
</div>
""", unsafe_allow_html=True)

# ── Load models (cached) — show file status if any file is bad ─
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

left, right = st.columns([1, 1], gap="large")

# ── LEFT: Inputs ───────────────────────────────────────────────
with left:
    st.markdown('<div class="sec-label">Soil Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload soil photo (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # Read bytes immediately — before any other widget touches the stream
    img_bytes = uploaded.read() if uploaded else None

    if img_bytes:
        st.image(io.BytesIO(img_bytes), use_container_width=True)

    st.markdown('<div class="sec-label">Soil & Climate Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        n    = st.number_input("Nitrogen — N",     0.0, 200.0,  90.0, step=1.0)
        temp = st.number_input("Temperature °C",   0.0,  50.0,  25.0, step=0.1)
        ph   = st.number_input("Soil pH",          3.0,  10.0,   6.5, step=0.01)
    with c2:
        p    = st.number_input("Phosphorus — P",   0.0, 200.0,  42.0, step=1.0)
        hum  = st.number_input("Humidity %",       0.0, 100.0,  80.0, step=1.0)
        yld  = st.number_input("Yield Last Season",0.0,15000.0,2500.0, step=10.0)
    with c3:
        k    = st.number_input("Potassium — K",    0.0, 200.0,  43.0, step=1.0)
        rain = st.number_input("Rainfall mm",      0.0,3000.0, 200.0, step=5.0)
        fert = st.number_input("Fertilizer Used",  0.0,1000.0, 120.0, step=5.0)

    st.markdown('<div class="sec-label">Farm Context</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        prev   = st.selectbox("Previous Crop",
                               ["Wheat","Rice","Maize","Cotton","Potato","Sugarcane","Tomato"])
    with d2:
        irrig  = st.selectbox("Irrigation", ["Canal","Drip","Rainfed","Sprinkler"])
        region = st.selectbox("Region", ["South","North","East","West","Central"])

    predict_clicked = st.button("Analyze Soil")

# ── RIGHT: Results ─────────────────────────────────────────────
with right:
    st.markdown('<div class="sec-label">Analysis Results</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="placeholder">
          <p><strong>Upload a soil image</strong> and fill in the parameters,<br>
             then click <strong>Analyze Soil</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)

    elif not img_bytes:
        st.error("Please upload a soil image before analyzing.")

    else:
        with st.spinner("Running inference…"):
            try:
                soil_name, confidence, all_probs, soil_fert, crop_recs, dbg = run_inference(
                    img_model, tab_proj, fusion, xgb_clf, scaler,
                    CLASS_NAMES, NUMERIC_COLS,
                    img_bytes,                          # raw bytes — no pointer issue
                    n, p, k, temp, hum, rain, ph, yld, fert,
                    season, irrig, prev, region,
                )

                # ── Soil type ──────────────────────────────────
                st.markdown(f"""
                <div class="soil-result">
                  <div class="label">Detected Soil Type</div>
                  <div class="name">{soil_name}</div>
                  <span class="conf">Confidence: {confidence}%</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability breakdown ──────────────────────
                st.markdown("**Soil Probabilities**")
                for name, pct in sorted(all_probs.items(), key=lambda x: -x[1]):
                    fill = int(pct)
                    st.markdown(f"""
                    <div class="prob-row">
                      <span class="prob-label">{name}</span>
                      <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{fill}%"></div>
                      </div>
                      <span class="prob-pct">{pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Top 3 crops ────────────────────────────────
                st.markdown("**Recommended Crops**")
                badge_cls = ["", "s", "t"]
                for crop in crop_recs:
                    stars  = "★" * crop["stars"] + "☆" * (5 - crop["stars"])
                    is_top = "top" if crop["rank"] == 1 else ""
                    bc     = badge_cls[crop["rank"] - 1]
                    st.markdown(f"""
                    <div class="crop-card {is_top}">
                      <div class="crop-rank-badge {bc}">#{crop["rank"]}</div>
                      <div class="crop-info" style="flex:1">
                        <div class="crop-name">{crop["name"]}</div>
                        <div class="crop-sub">{crop["fertilizer"]} &nbsp;|&nbsp; {crop["npk"]}</div>
                      </div>
                      <div class="stars">{stars}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Soil fertilizer ────────────────────────────
                st.markdown(f"""
                <div class="fert-box">
                  <div class="fert-title">Soil Fertilizer Recommendation</div>
                  <div class="fert-name">{soil_fert["fertilizer"]}</div>
                  <div class="fert-npk">{soil_fert["npk"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Debug expander ─────────────────────────────
                with st.expander("Prediction Debug"):
                    st.write("Raw probs:", dbg["probs"])
                    st.write("Image feat std:", dbg["img_feat_std"])

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.markdown("**File status:**")
                for fn, info in _file_status.items():
                    st.write(f"- `{fn}`: {info['mb']:.1f} MB")
