from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import torchreid  # type: ignore
    HAS_TORCHREID = True
except Exception:
    HAS_TORCHREID = False


def build_transform(size: int = 256):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(arch: str, weights: Optional[str], device: str | None) -> Tuple[object, bool, "torch.device" | None]:
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available; reid_filter requires torch.")
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if HAS_TORCHREID and arch.startswith('osnet'):
        model = torchreid.models.build_model(name=arch, num_classes=1, pretrained=True if not weights else False)
        if weights and Path(weights).exists():
            sd = torch.load(weights, map_location='cpu')
            model.load_state_dict(sd, strict=False)
        model.eval().to(dev)
        return model, True, dev
    # Fallback to torchvision ResNet50 features
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(dev)
    return model, False, dev


def embed_images(model, is_reid: bool, img_paths: List[Path], device: "torch.device", size: int = 256, batch: int = 64) -> np.ndarray:
    tfm = build_transform(size)
    all_feats: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(img_paths), batch):
            batch_imgs = []
            for p in img_paths[i:i+batch]:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    continue
                batch_imgs.append(tfm(img))
            if not batch_imgs:
                continue
            x = torch.stack(batch_imgs, dim=0).to(device)
            feats = model(x)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu().numpy())
    if not all_feats:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(all_feats, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def score_against_gallery(gallery_dir: str, query_paths: List[str], arch: str = 'osnet_x1_0', weights: Optional[str] = None,
                          size: int = 256, batch: int = 64, device: Optional[str] = None) -> List[Tuple[str, float, Optional[str]]]:
    """Return list of (query_path, max_similarity, best_gallery_path)"""
    g_files = [p for p in Path(gallery_dir).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    if not g_files:
        return [(qp, 0.0, None) for qp in query_paths]
    model, is_reid, dev = load_model(arch, weights, device)
    g_embs = embed_images(model, is_reid, g_files, dev, size=size, batch=batch)
    q_files = [Path(qp) for qp in query_paths]
    q_embs = embed_images(model, is_reid, q_files, dev, size=size, batch=batch)
    if q_embs.shape[0] == 0 or g_embs.shape[0] == 0:
        return [(qp, 0.0, None) for qp in query_paths]
    sims = cosine_sim(q_embs, g_embs)
    max_idx = sims.argmax(axis=1)
    max_sims = sims.max(axis=1)
    out: List[Tuple[str, float, Optional[str]]] = []
    for i, qp in enumerate(query_paths):
        gi = int(max_idx[i])
        out.append((qp, float(max_sims[i]), str(g_files[gi])))
    return out
