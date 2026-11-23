import glob, json, os, numpy as np, torch, torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from ensemble_boxes import weighted_boxes_fusion


class NoduleDetector:
    """Ensemble of Faster R-CNN v2 fold models, fused with Weighted Boxes Fusion."""

    def __init__(self, model_dir="models"):
        cfg = json.load(open(os.path.join(model_dir, "ensemble_config.json")))
        self.single = cfg["single_class"]
        self.default_thr = 0.15
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for p in sorted(glob.glob(os.path.join(model_dir, "frcnn_f*.pt"))):
            m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=800, max_size=800)
            m.rpn.anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2 if self.single else 3)
            m.roi_heads.detections_per_img = 50
            m.roi_heads.score_thresh = 0.001
            m.load_state_dict(torch.load(p, map_location=self.dev))
            self.models.append(m.to(self.dev).eval())
        if not self.models:
            raise FileNotFoundError(f"No frcnn_f*.pt found in '{model_dir}'")

    @torch.no_grad()
    def predict(self, img, thr=None):
        """Returns a list of {'box': [x1,y1,x2,y2], 'score': float, 'label': int} in pixel coordinates."""
        thr = self.default_thr if thr is None else thr
        img = img.convert("RGB"); W, H = img.size; sc = np.array([W, H, W, H])
        t = TF.to_tensor(img).to(self.dev)
        preds = []
        for m in self.models:
            p = m([t])[0]
            preds.append((p["boxes"].cpu().numpy(), p["scores"].cpu().numpy(), p["labels"].cpu().numpy()))
        b, s, l = weighted_boxes_fusion(
            [np.clip(p[0] / sc, 0, 1).tolist() for p in preds],
            [p[1].tolist() for p in preds],
            [p[2].tolist() for p in preds],
            iou_thr=0.5, skip_box_thr=0.001)
        out = [{"box": (bb * sc).tolist(), "score": float(ss), "label": int(ll)} for bb, ss, ll in zip(b, s, l) if ss >= thr]
        return sorted(out, key=lambda d: -d["score"])
