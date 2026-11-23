import io, pandas as pd, streamlit as st
from PIL import Image, ImageDraw
from nodule_detector import NoduleDetector, MODES

st.set_page_config(page_title="Lung Nodule Detection", layout="wide")
st.title("Lung Nodule Detection in CT Images")
st.caption("Cross-architecture ensemble: RF-DETR (DINOv2 ViT) + Faster R-CNN v2 (ResNet50-FPN), fused with Weighted Boxes Fusion. Research demo, not a medical device.")

@st.cache_resource(show_spinner="Loading 10 models (first time only)...")
def load(): return NoduleDetector("models")
det = load()

@st.cache_data(show_spinner="Running ensemble...")
def run(img_bytes, mode): return det.predict(Image.open(io.BytesIO(img_bytes)), mode)

with st.sidebar:
    mode = st.selectbox("Inference mode", list(MODES))
    op = st.radio("Operating point", ["Balanced (best F1)", "Low false positives"])
    default = det.thr_balanced if op.startswith("Balanced") else det.thr_lowfp
    thr = st.slider("Confidence threshold", 0.05, 0.95, float(default), 0.025,
                    help="Lower = more nodules found but more false alarms. Higher = fewer false alarms but more missed nodules.")
    label_file = st.file_uploader("Optional: ground-truth label (.txt, YOLO format)", type=["txt"])

up = st.file_uploader("Upload a CT slice", type=["jpg", "jpeg", "png"])
if up:
    data = up.getvalue(); img = Image.open(io.BytesIO(data)).convert("RGB"); W, H = img.size
    dets = [k for k in run(data, mode) if k["score"] >= thr]
    vis = img.copy(); d = ImageDraw.Draw(vis); gt = []
    if label_file:
        for ln in label_file.getvalue().decode().strip().splitlines():
            _, x, y, w, h = map(float, ln.split()[:5]); gt.append([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])
        for b in gt: d.rectangle(b, outline=(0, 255, 0), width=2)
    for k in dets:
        b = k["box"]; d.rectangle(b, outline=(255, 0, 0), width=2); d.text((b[0], max(b[1]-12, 0)), f"{k['score']:.2f}", fill=(255, 0, 0))
    c1, c2 = st.columns(2)
    c1.image(img, caption="Input", use_container_width=True)
    c2.image(vis, caption="Prediction (red)" + (" vs ground truth (green)" if gt else ""), use_container_width=True)
    st.subheader(f"{len(dets)} nodule(s) detected at threshold {thr:.3f}")
    if dets:
        st.dataframe(pd.DataFrame([{"#": i+1, "confidence": round(k["score"], 3), "x1": int(k["box"][0]), "y1": int(k["box"][1]),
                                    "x2": int(k["box"][2]), "y2": int(k["box"][3])} for i, k in enumerate(dets)]), hide_index=True)
