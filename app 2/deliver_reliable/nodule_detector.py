import glob, json, os
import numpy as np, torch, torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from ensemble_boxes import weighted_boxes_fusion
from rfdetr import RFDETRMedium

TTA4 = ("none", "h", "v", "hv")
MODES = {
    "Full consensus (10 models + 4-way TTA) - most accurate, slowest": ("both", TTA4),
    "Consensus, no TTA (10 models) - faster": ("both", ("none",)),
    "RF-DETR only (5 models, no TTA) - fastest": ("rfdetr", ("none",)),
}


def _flip(img, m):
    return ImageOps.mirror(img) if m == "h" else ImageOps.flip(img) if m == "v" else ImageOps.flip(ImageOps.mirror(img)) if m == "hv" else img


def _unflip(b, W, H, m):
    b = b.copy()
    if len(b) == 0: return b
    if m in ("h", "hv"): b[:, [0, 2]] = W - b[:, [2, 0]]
    if m in ("v", "hv"): b[:, [1, 3]] = H - b[:, [3, 1]]
    return b


class NoduleDetector:
    """Cross-architecture ensemble: 5x RF-DETR Medium + 5x Faster R-CNN v2, fused with Weighted Boxes Fusion."""

    def __init__(self, model_dir="models"):
        self.cfg = json.load(open(os.path.join(model_dir, "results", "ensemble_config.json")))
        self.single = self.cfg["single_class"]; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.w_r, self.w_f = self.cfg["wbf_weights_rfdetr_frcnn"]
        self.thr_balanced, self.thr_lowfp = self.cfg["thr_f1_opt"], self.cfg["thr_low_fp"]

        self.frcnn = []
        for p in sorted(glob.glob(os.path.join(model_dir, "frcnn", "frcnn_tiny_f*.pt"))):
            size = self.cfg["frcnn_input_size"]
            m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=size, max_size=size)
            sizes = ((12, 16, 24), (24, 32, 48), (48, 64, 96), (96, 128, 160), (192, 256, 320))
            m.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((0.7, 1.0, 1.4),) * 5)
            m.rpn.head = RPNHead(m.backbone.out_channels, m.rpn.anchor_generator.num_anchors_per_location()[0], conv_depth=2)
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2 if self.single else 3)
            m.roi_heads.detections_per_img = 30; m.roi_heads.score_thresh = 0.01; m.roi_heads.nms_thresh = 0.45
            m.load_state_dict(torch.load(p, map_location=self.dev)); self.frcnn.append(m.to(self.dev).eval())

        self.rfdetr = [RFDETRMedium(pretrain_weights=p, resolution=self.cfg["rfdetr_resolution"])
                       for p in sorted(glob.glob(os.path.join(model_dir, "rfdetr", "rfdetr_f*.pth")))]
        if not self.frcnn or not self.rfdetr:
            raise FileNotFoundError(f"Missing models in '{model_dir}' (found {len(self.frcnn)} FRCNN, {len(self.rfdetr)} RF-DETR)")

    @torch.no_grad()
    def _frcnn_raw(self, m, img, mode):
        W, H = img.size; out = m([TF.to_tensor(_flip(img, mode)).to(self.dev)])[0]
        return _unflip(out["boxes"].cpu().numpy(), W, H, mode), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()

    def _rfdetr_raw(self, m, img, mode):
        W, H = img.size; d = m.predict(_flip(img, mode), threshold=0.01); d = d[0] if isinstance(d, list) else d
        if len(d.xyxy) == 0: return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        b = _unflip(np.asarray(d.xyxy, np.float32), W, H, mode)
        lab = np.ones(len(b), np.int64) if self.single else np.asarray(d.class_id, np.int64) + 1
        return b, np.asarray(d.confidence, np.float32), lab

    def predict(self, img, mode_name):
        """Returns ALL fused detections (score >= 0.01), sorted by score. Apply the confidence threshold afterwards."""
        which, tta = MODES[mode_name]; img = img.convert("RGB"); W, H = img.size; sc = np.array([W, H, W, H], np.float32)
        specs = [("r", m) for m in self.rfdetr] + ([("f", m) for m in self.frcnn] if which == "both" else [])
        preds, weights = [], []
        for kind, m in specs:
            for t in tta:
                preds.append(self._rfdetr_raw(m, img, t) if kind == "r" else self._frcnn_raw(m, img, t))
                weights.append(self.w_r if kind == "r" else self.w_f)
        if len(preds) == 1:
            b, s = preds[0][0], preds[0][1]
        else:
            bl = [np.clip(p[0] / sc, 0, 1).tolist() if len(p[0]) else [] for p in preds]
            sl = [p[1].tolist() for p in preds]; ll = [p[2].tolist() for p in preds]
            b, s, _ = weighted_boxes_fusion(bl, sl, ll, weights=weights if which == "both" else None, iou_thr=0.45, skip_box_thr=0.01)
            b = np.asarray(b) * sc if len(b) else np.zeros((0, 4))
        out = [{"box": [float(v) for v in bb], "score": float(ss)} for bb, ss in zip(b, s)]
        return sorted(out, key=lambda d: -d["score"])
