# api.py — Flask REST API for Soil Advisor mobile app
# Run: python api.py
# Listens on port 8000

import io, os, json, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
def mpath(n): return os.path.join(BASE, n)

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
# MODEL LOADING — done once at startup
# ══════════════════════════════════════════════════════════════

print("Loading models...")
with open(mpath("model_config.json")) as f: cfg = json.load(f)
with open(mpath("class_names.json"))  as f: CLASS_NAMES = json.load(f)

img_dim   = cfg["IMG_FEAT_DIM"]
xgb_dim   = cfg["XGB_PROJ_DIM"]
fused_dim = cfg["FUSED_DIM"]
num_heads = cfg["NUM_HEADS"]
num_cls   = cfg["NUM_CLASSES"]
tab_dim   = cfg["TAB_FEAT_DIM"]
NUMERIC_COLS = cfg["NUMERIC_COLS"]

img_model = ResNet50Classifier(num_cls, img_dim)
tab_proj  = TabProjector(tab_dim, xgb_dim)
fusion    = FusionGRNModel(img_dim, xgb_dim, fused_dim, num_heads, num_cls)

img_model.load_state_dict(torch.load(mpath("img_model.pt"),     map_location="cpu", weights_only=True))
tab_proj.load_state_dict(torch.load(mpath("tab_projector.pt"), map_location="cpu", weights_only=True))
fusion.load_state_dict(torch.load(mpath("fusion_model.pt"),    map_location="cpu", weights_only=True))

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(mpath("xgb_model.json"))

with open(mpath("scaler.pkl"), "rb") as fh:
    scaler = pickle.load(fh)

img_model.eval(); tab_proj.eval(); fusion.eval()
print("Models loaded OK.")

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
    ("Red Soil",      "Kharif"): ["Cotton",    "Maize",      "Groundnut", "Tomato"],
    ("Red Soil",      "Rabi")  : ["Wheat",     "Sunflower",  "Linseed",   "Potato"],
    ("Red Soil",      "Zaid")  : ["Watermelon","Cucumber",   "Bitter Gourd","Moong"],
    ("Alluvial Soil", "Kharif"): ["Rice",      "Sugarcane",  "Maize",     "Jute"],
    ("Alluvial Soil", "Rabi")  : ["Wheat",     "Mustard",    "Barley",    "Peas"],
    ("Alluvial Soil", "Zaid")  : ["Watermelon","Muskmelon",  "Cucumber",  "Moong"],
    ("Black Soil",    "Kharif"): ["Cotton",    "Sorghum",    "Soybean",   "Groundnut"],
    ("Black Soil",    "Rabi")  : ["Wheat",     "Chickpea",   "Linseed",   "Safflower"],
    ("Black Soil",    "Zaid")  : ["Sunflower", "Sesame",     "Maize",     "Moong"],
    ("Clay Soil",     "Kharif"): ["Rice",      "Jute",       "Sugarcane", "Taro"],
    ("Clay Soil",     "Rabi")  : ["Wheat",     "Barley",     "Mustard",   "Spinach"],
    ("Clay Soil",     "Zaid")  : ["Cucumber",  "Bitter Gourd","Pumpkin",  "Moong"],
    ("Laterite Soil", "Kharif"): ["Cashew",    "Rubber",     "Tea",       "Coffee"],
    ("Laterite Soil", "Rabi")  : ["Tapioca",   "Groundnut",  "Turmeric",  "Ginger"],
    ("Laterite Soil", "Zaid")  : ["Mango",     "Pineapple",  "Jackfruit", "Banana"],
    ("Yellow Soil",   "Kharif"): ["Rice",      "Maize",      "Groundnut", "Sesame"],
    ("Yellow Soil",   "Rabi")  : ["Wheat",     "Mustard",    "Potato",    "Barley"],
    ("Yellow Soil",   "Zaid")  : ["Sunflower", "Moong",      "Cucumber",  "Tomato"],
}

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        img_file = request.files["image"]
        img_bytes = img_file.read()

        # Parse form fields
        def gf(key, default=0.0):
            return float(request.form.get(key, default))

        n      = gf("n",    90.0)
        p      = gf("p",    42.0)
        k      = gf("k",    43.0)
        temp   = gf("temp", 25.0)
        hum    = gf("hum",  80.0)
        rain   = gf("rain", 200.0)
        ph     = gf("ph",   6.5)
        yld    = gf("yld",  2500.0)
        fert   = gf("fert", 120.0)
        season = request.form.get("season", "Kharif")
        irrig  = request.form.get("irrig",  "Canal")
        prev   = request.form.get("prev",   "Wheat")
        region = request.form.get("region", "South")

        # Image transform
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_t   = tf(pil_img).unsqueeze(0)

        # Tabular features
        num_raw   = np.array([[n, p, k, temp, hum, rain, ph, yld, fert]])
        num_sc    = scaler.transform(pd.DataFrame(num_raw, columns=NUMERIC_COLS))
        cat_enc   = np.array([[SEASON_MAP[season], IRRIG_MAP[irrig],
                               PREV_MAP[prev], REGION_MAP[region]]])
        xgb_input = np.concatenate([num_sc, cat_enc], axis=1).astype(np.float32)
        xgb_probs = xgb_clf.predict_proba(xgb_input)
        tab_raw   = np.concatenate([xgb_probs, xgb_input], axis=1).astype(np.float32)
        tab_t     = torch.tensor(tab_raw, dtype=torch.float32)

        # Inference
        img_model.eval(); tab_proj.eval(); fusion.eval()
        with torch.no_grad():
            img_feat  = img_model(img_t, return_features=True)
            tab_feat  = tab_proj(tab_t)
            logits, _ = fusion(img_feat, tab_feat)

        probs      = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx   = int(np.argmax(probs))
        soil_name  = CLASS_NAMES[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)
        all_probs  = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                      for i in range(len(CLASS_NAMES))}

        soil_fert = SOIL_FERT_MAP.get(soil_name,
                    {"fertilizer": "NPK 14:14:14", "npk": "N:P:K = 60:30:30 kg/ha"})

        crops_all = CROP_MAP.get(
            (soil_name, season),
            CROP_MAP.get((soil_name, "Kharif"), ["Wheat", "Rice", "Maize"]))

        crop_recs = []
        for i, crop in enumerate(crops_all[:3]):
            cf = CROP_FERT_MAP.get(crop, {"fertilizer": "NPK 14:14:14", "npk": "60:40:20 kg/ha"})
            crop_recs.append({
                "name":       crop,
                "rank":       i + 1,
                "stars":      3 - i,
                "fertilizer": cf["fertilizer"],
                "npk":        cf["npk"],
            })

        return jsonify({
            "soil_name":  soil_name,
            "confidence": confidence,
            "all_probs":  all_probs,
            "soil_fert":  soil_fert,
            "crop_recs":  crop_recs,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
