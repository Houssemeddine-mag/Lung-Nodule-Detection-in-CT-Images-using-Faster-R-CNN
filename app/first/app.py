import pandas as pd, streamlit as st
from PIL import Image, ImageDraw
from nodule_detector import NoduleDetector

st.set_page_config(page_title="Lung Nodule Detection", layout="wide")
st.title("Lung Nodule Detection in CT Images")
st.caption("Ensemble of 5 Faster R-CNN v2 models (ResNet50-FPN). Research demo, not a medical device.")

@st.cache_resource
def load(): return NoduleDetector("models")
det = load()

with st.sidebar:
    thr = st.slider("Confidence threshold", 0.01, 0.90, det.default_thr, 0.01,
                    help="Lower = more nodules found but more false alarms. Higher = fewer false alarms but more missed nodules.")
    label_file = st.file_uploader("Optional: ground-truth label (.txt, YOLO format)", type=["txt"])

up = st.file_uploader("Upload a CT slice", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up).convert("RGB"); W, H = img.size
    dets = det.predict(img, thr)
    vis = img.copy(); d = ImageDraw.Draw(vis)
    gt = []
    if label_file:
        for ln in label_file.getvalue().decode().strip().splitlines():
            _, x, y, w, h = map(float, ln.split()[:5]); gt.append([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])
        for b in gt: d.rectangle(b, outline=(0, 255, 0), width=2)
    for k in dets:
        b = k["box"]; d.rectangle(b, outline=(255, 0, 0), width=2); d.text((b[0], max(b[1]-12, 0)), f"{k['score']:.2f}", fill=(255, 0, 0))
    c1, c2 = st.columns(2)
    c1.image(img, caption="Input", use_container_width=True)
    c2.image(vis, caption="Prediction (red)" + (" vs ground truth (green)" if gt else ""), use_container_width=True)
    st.subheader(f"{len(dets)} nodule(s) detected at threshold {thr:.2f}")
    if dets:
        st.dataframe(pd.DataFrame([{"#": i+1, "confidence": round(k["score"], 3), "x1": int(k["box"][0]), "y1": int(k["box"][1]),
                                    "x2": int(k["box"][2]), "y2": int(k["box"][3])} for i, k in enumerate(dets)]), hide_index=True)
