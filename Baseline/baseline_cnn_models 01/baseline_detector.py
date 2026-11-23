import glob, os
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision.ops import nms
from ensemble_boxes import weighted_boxes_fusion

IMG, STRIDE = 416, 16
G = IMG // STRIDE
BOX_SCALE = 0.15 * IMG


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class BaselineCNN(nn.Module):
    """Plain CNN: 4 conv blocks + 1 conv + 1x1 head -> 26x26 grid of (objectness, x, y, w, h)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                                      nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(128, 5, 1)

    def forward(self, x):
        return self.head(self.features(x))


class BaselineDetector:
    def __init__(self, model_dir="models"):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.models = []
        for p in sorted(glob.glob(os.path.join(model_dir, "baseline_cnn_f*.pt"))):
            m = BaselineCNN(); m.load_state_dict(torch.load(p, map_location=self.dev)); self.models.append(m.to(self.dev).eval())
        if not self.models:
            raise FileNotFoundError(f"No baseline_cnn_f*.pt found in '{model_dir}'")

    @torch.no_grad()
    def _single(self, m, x):
        out = m(x)[0]; obj = torch.sigmoid(out[0]); reg = torch.sigmoid(out[1:])
        gi, gj = (obj > 0.01).nonzero(as_tuple=True); s = obj[gi, gj]
        cx = (gj + reg[0][gi, gj]) * STRIDE; cy = (gi + reg[1][gi, gj]) * STRIDE
        w = reg[2][gi, gj] * BOX_SCALE; h = reg[3][gi, gj] * BOX_SCALE
        b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clamp(0, IMG)
        keep = nms(b, s, 0.5)[:30] if len(b) else torch.zeros(0, dtype=torch.long, device=b.device)
        return b[keep].cpu().numpy(), s[keep].cpu().numpy(), obj.cpu().numpy()

    def predict(self, img, which="ensemble"):
        """which: model index (int) or 'ensemble'. Returns (detections in original pixels, 26x26 objectness heatmap)."""
        img = img.convert("RGB"); W, H = img.size
        x = torch.from_numpy(np.asarray(img.resize((IMG, IMG)), dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - 0.5) / 0.5)[None].to(self.dev)
        models = self.models if which == "ensemble" else [self.models[which]]
        outs = [self._single(m, x) for m in models]; heat = np.mean([o[2] for o in outs], 0)
        if len(outs) == 1:
            b, s = outs[0][0], outs[0][1]
        else:
            b, s, _ = weighted_boxes_fusion([np.clip(o[0] / IMG, 0, 1).tolist() for o in outs], [o[1].tolist() for o in outs],
                                            [[0] * len(o[1]) for o in outs], iou_thr=0.45, skip_box_thr=0.01)
            b = np.asarray(b) * IMG if len(b) else np.zeros((0, 4))
        sc = np.array([W / IMG, H / IMG, W / IMG, H / IMG])
        dets = [{"box": [float(v) for v in bb * sc], "score": float(ss)} for bb, ss in zip(b, s)]
        return sorted(dets, key=lambda d: -d["score"]), heat
