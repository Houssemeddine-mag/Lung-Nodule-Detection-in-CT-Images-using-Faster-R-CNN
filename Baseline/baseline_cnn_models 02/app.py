import io, numpy as np, pandas as pd, streamlit as st
from PIL import Image, ImageChops, ImageDraw
from baseline_detector import BaselineDetector

DEFAULT_THR = 0.30   # <-- replace with the "Tuned threshold" printed in your baseline notebook (section 7)

st.set_page_config(page_title="Baseline CNN - Lung Nodule Detection", layout="wide")
st.title("Baseline CNN: Lung Nodule Detection")
st.caption("Plain CNN trained from scratch (about 0.2 M parameters, 26x26 detection grid). Baseline for comparison, not a medical device.")

@st.cache_resource(show_spinner="Loading models...")
def load(): return BaselineDetector("models")
det = load()

@st.cache_data(show_spinner="Running CNN...")
def run(img_bytes, which): return det.predict(Image.open(io.BytesIO(img_bytes)), which)

with st.sidebar:
    n = len(det.models)
    choice = st.selectbox("Model", ["Ensemble of all folds"] + [f"Fold {i} model" for i in range(n)])
    which = "ensemble" if choice.startswith("Ensemble") else int(choice.split()[1])
    thr = st.slider("Confidence threshold", 0.02, 0.95, DEFAULT_THR, 0.01,
                    help="Lower = more nodules found but more false alarms. Higher = fewer false alarms but more missed nodules.")
    show_heat = st.checkbox("Show objectness heatmap (what the CNN 'sees')", value=True)
    label_file = st.file_uploader("Optional: ground-truth label (.txt, YOLO format)", type=["txt"])

up = st.file_uploader("Upload a CT slice", type=["jpg", "jpeg", "png"])
if up:
    data = up.getvalue(); img = Image.open(io.BytesIO(data)).convert("RGB"); W, H = img.size
    all_dets, heat = run(data, which); dets = [k for k in all_dets if k["score"] >= thr]
    vis = img.copy(); d = ImageDraw.Draw(vis); gt = []
    if label_file:
        for ln in label_file.getvalue().decode().strip().splitlines():
            _, x, y, w, h = map(float, ln.split()[:5]); gt.append([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])
        for b in gt: d.rectangle(b, outline=(0, 255, 0), width=2)
    for k in dets:
        b = k["box"]; d.rectangle(b, outline=(255, 0, 0), width=2); d.text((b[0], max(b[1]-12, 0)), f"{k['score']:.2f}", fill=(255, 0, 0))
    cols = st.columns(3 if show_heat else 2)
    cols[0].image(img, caption="Input", use_container_width=True)
    cols[1].image(vis, caption="Prediction (red)" + (" vs ground truth (green)" if gt else ""), use_container_width=True)
    if show_heat:
        h8 = Image.fromarray((np.clip(heat / max(heat.max(), 1e-6), 0, 1) * 255).astype("uint8")).resize((W, H), Image.BILINEAR)
        red = Image.merge("RGB", (h8, Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)))
        cols[2].image(ImageChops.screen(img, red), caption="Objectness heatmap (brighter red = more nodule-like)", use_container_width=True)
    st.subheader(f"{len(dets)} nodule(s) detected at threshold {thr:.2f}")
    if dets:
        st.dataframe(pd.DataFrame([{"#": i+1, "confidence": round(k["score"], 3), "x1": int(k["box"][0]), "y1": int(k["box"][1]),
                                    "x2": int(k["box"][2]), "y2": int(k["box"][3])} for i, k in enumerate(dets)]), hide_index=True)
